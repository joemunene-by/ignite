import os

import pytest
import torch
from sklearn.metrics import r2_score

import ignite.distributed as idist
from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics.regression import R2Score


def test_zero_sample():
    m = R2Score()
    with pytest.raises(NotComputableError, match=r"R2Score must have at least one example before it can be computed"):
        m.compute()


def test_wrong_input_shapes():
    m = R2Score()

    with pytest.raises(ValueError, match=r"Input data shapes should be the same, but given"):
        m.update((torch.rand(4), torch.rand(4, 1)))

    with pytest.raises(ValueError, match=r"Input data shapes should be the same, but given"):
        m.update((torch.rand(4, 1), torch.rand(4)))


def test_r2_score(available_device):
    torch.manual_seed(42)
    size = 51

    y_pred = torch.rand(size)
    y = torch.rand(size)

    m = R2Score(device=available_device)
    assert m._device == torch.device(available_device)

    m.reset()
    m.update((y_pred, y))

    expected = r2_score(y.cpu().numpy(), y_pred.cpu().numpy())
    assert m.compute() == pytest.approx(expected)


def test_r2_score_2(available_device):
    torch.manual_seed(1)
    size = 105
    y_pred = torch.rand(size, 1)
    y = torch.rand(size, 1)

    y = y[torch.randperm(size)]

    m = R2Score(device=available_device)
    assert m._device == torch.device(available_device)

    m.reset()
    batch_size = 16
    n_iters = size // batch_size + 1
    for i in range(n_iters):
        idx = i * batch_size
        m.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))

    expected = r2_score(y.cpu().numpy(), y_pred.cpu().numpy())
    assert m.compute() == pytest.approx(expected)


def test_r2_score_numerical_stability_under_large_mean():
    """Regression for the Welford-based denominator.

    Constructs a float32 ``y`` with large mean (1e6) and small relative
    variance. The naive ``Σ y² − (Σ y)² / n`` denominator catastrophically
    cancels at this scale and the resulting R² is way off; the Welford
    denominator preserves the relevant low-order bits and the metric
    stays within sklearn's float64 reference.
    """
    torch.manual_seed(0)
    n = 1024
    base = torch.full((n,), 1e6, dtype=torch.float32)
    y_f32 = base + torch.randn(n, dtype=torch.float32)
    y_pred_f32 = y_f32 + 0.1 * torch.randn(n, dtype=torch.float32)

    # sklearn reference in float64 land.
    expected = r2_score(y_f32.to(torch.float64).numpy(), y_pred_f32.to(torch.float64).numpy())

    m = R2Score()
    m.update((y_pred_f32, y_f32))
    got = m.compute()

    assert got == pytest.approx(expected, rel=1e-6, abs=1e-6)

    # Sanity check: the naive float32 formula would not get this close.
    # If a future change re-introduced it, this assertion would still
    # protect the contract because we compare against the float64 ref.
    naive_denom = (y_f32 * y_f32).sum() - (y_f32.sum() ** 2) / n
    naive_num = ((y_pred_f32 - y_f32) ** 2).sum()
    naive_r2 = (1 - naive_num / naive_denom).item()
    # On float32 with mean 1e6 the naive denominator collapses to noise
    # and the resulting R² drifts well outside the 1e-6 band the
    # Welford path holds.
    assert abs(naive_r2 - expected) > 1e-3, (
        "Test setup must produce a regime where the naive formula fails; "
        "if this assertion fails, pick a larger mean or smaller variance."
    )


def test_r2_score_zero_variance_y_raises():
    """R² is undefined when ``y`` has zero variance. The original code
    silently returned ``-inf`` or ``nan`` in this case; the Welford
    port raises ``NotComputableError`` so a caller cannot accidentally
    feed garbage into a downstream pipeline.
    """
    m = R2Score()
    y = torch.full((16,), 3.0)
    y_pred = torch.randn(16)
    m.update((y_pred, y))
    with pytest.raises(NotComputableError, match=r"zero variance"):
        m.compute()


def test_integration_r2_score(available_device):
    torch.manual_seed(1)
    size = 105
    y_pred = torch.rand(size, 1)
    y = torch.rand(size, 1)

    # Shuffle targets
    y = y[torch.randperm(size)]

    batch_size = 15

    def update_fn(engine, batch):
        idx = (engine.state.iteration - 1) * batch_size
        return y_pred[idx : idx + batch_size], y[idx : idx + batch_size]

    engine = Engine(update_fn)

    m = R2Score(device=available_device)
    assert m._device == torch.device(available_device)
    m.attach(engine, "r2_score")

    data = list(range(size // batch_size))
    r_squared = engine.run(data, max_epochs=1).metrics["r2_score"]

    expected = r2_score(y.cpu().numpy(), y_pred.cpu().numpy())
    assert r_squared == pytest.approx(expected)


def _test_distrib_compute(device, tol=1e-6):
    rank = idist.get_rank()

    def _test(metric_device):
        metric_device = torch.device(metric_device)
        m = R2Score(device=metric_device)

        y_pred = torch.randint(0, 10, size=(10,), device=device).float()
        y = torch.randint(0, 10, size=(10,), device=device).float()

        m.update((y_pred, y))

        # gather y_pred, y
        y_pred = idist.all_gather(y_pred)
        y = idist.all_gather(y)

        np_y_pred = y_pred.cpu().numpy()
        np_y = y.cpu().numpy()
        res = m.compute()
        assert r2_score(np_y, np_y_pred) == pytest.approx(res, abs=tol)

    for i in range(3):
        torch.manual_seed(10 + rank + i)
        _test("cpu")
        if device.type != "xla":
            _test(idist.device())


def _test_distrib_integration(device):
    rank = idist.get_rank()

    def _test(n_epochs, metric_device):
        metric_device = torch.device(metric_device)
        n_iters = 80
        batch_size = 16

        y_true = torch.randint(0, 10, size=(n_iters * batch_size,)).to(device).float()
        y_preds = torch.randint(0, 10, size=(n_iters * batch_size,)).to(device).float()

        def update(engine, i):
            return (
                y_preds[i * batch_size : (i + 1) * batch_size],
                y_true[i * batch_size : (i + 1) * batch_size],
            )

        engine = Engine(update)

        r2 = R2Score(device=metric_device)
        r2.attach(engine, "r2")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=n_epochs)

        y_preds = idist.all_gather(y_preds)
        y_true = idist.all_gather(y_true)

        assert "r2" in engine.state.metrics

        res = engine.state.metrics["r2"]
        if isinstance(res, torch.Tensor):
            res = res.cpu().numpy()

        true_res = r2_score(y_true.cpu().numpy(), y_preds.cpu().numpy())

        assert pytest.approx(res) == true_res

    metric_devices = ["cpu"]
    if device.type != "xla":
        metric_devices.append(idist.device())
    for metric_device in metric_devices:
        for i in range(2):
            torch.manual_seed(12 + rank + i)
            _test(n_epochs=1, metric_device=metric_device)
            _test(n_epochs=2, metric_device=metric_device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_nccl_gpu(distributed_context_single_node_nccl):
    device = idist.device()
    _test_distrib_compute(device)
    _test_distrib_integration(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_gloo_cpu_or_gpu(distributed_context_single_node_gloo):
    device = idist.device()
    _test_distrib_compute(device)
    _test_distrib_integration(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_distrib_hvd(gloo_hvd_executor):
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    nproc = 4 if not torch.cuda.is_available() else torch.cuda.device_count()

    gloo_hvd_executor(_test_distrib_compute, (device,), np=nproc, do_init=True)
    gloo_hvd_executor(_test_distrib_integration, (device,), np=nproc, do_init=True)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gloo_cpu_or_gpu(distributed_context_multi_node_gloo):
    device = idist.device()
    _test_distrib_compute(device)
    _test_distrib_integration(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_nccl_gpu(distributed_context_multi_node_nccl):
    device = idist.device()
    _test_distrib_compute(device)
    _test_distrib_integration(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_single_device_xla():
    device = idist.device()
    _test_distrib_compute(device, tol=1e-3)
    _test_distrib_integration(device)


def _test_distrib_xla_nprocs(index):
    device = idist.device()
    _test_distrib_compute(device, tol=1e-3)
    _test_distrib_integration(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_distrib_xla_nprocs, args=(), nprocs=n)
