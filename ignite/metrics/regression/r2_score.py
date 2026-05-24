import torch

import ignite.distributed as idist

from ignite.exceptions import NotComputableError
from ignite.metrics._running_stats import WelfordVariance
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce
from ignite.metrics.regression._base import _BaseRegression


class R2Score(_BaseRegression):
    r"""Calculates the R-Squared, the
    `coefficient of determination <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_.

    .. math::
        R^2 = 1 - \frac{\sum_{j=1}^n(A_j - P_j)^2}{\sum_{j=1}^n(A_j - \bar{A})^2}

    where :math:`A_j` is the ground truth, :math:`P_j` is the predicted value and
    :math:`\bar{A}` is the mean of the ground truth.

    The denominator :math:`\sum (A_j - \bar{A})^2` is accumulated via
    :class:`~ignite.metrics._running_stats.WelfordVariance` to avoid the
    catastrophic cancellation of the naive
    :math:`\sum A_j^2 - (\sum A_j)^2 / n` formula on large means.

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    - `y` and `y_pred` must be of same shape `(N, )` or `(N, 1)` and of type `float32`.

    Parameters are inherited from ``Metric.__init__``.

    Args:
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            By default, metrics require the output as ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.

    Examples:
        To use with ``Engine`` and ``process_function``, simply attach the metric instance to the engine.
        The output of the engine's ``process_function`` needs to be in format of
        ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y, ...}``.

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            metric = R2Score()
            metric.attach(default_evaluator, 'r2')
            y_true = torch.tensor([0., 1., 2., 3., 4., 5.])
            y_pred = y_true * 0.75
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics['r2'])

        .. testoutput::

            0.8035...

    .. versionchanged:: 0.4.3
        Works with DDP.

    .. versionchanged:: 0.5.3
        Denominator now uses :class:`~ignite.metrics._running_stats.WelfordVariance`
        for numerical stability. The metric also raises
        :class:`~ignite.exceptions.NotComputableError` when the ground truth has
        zero variance (previously returned ``-inf`` or ``nan``).
    """

    _state_dict_all_req_keys = ("_num_examples", "_sum_of_errors", "_y_running_stats")

    @reinit__is_reduced
    def reset(self) -> None:
        self._num_examples = 0
        self._sum_of_errors = torch.tensor(0.0, device=self._device)
        self._y_running_stats = WelfordVariance()

    def _update(self, output: tuple[torch.Tensor, torch.Tensor]) -> None:
        y_pred, y = output
        self._num_examples += y.shape[0]
        self._sum_of_errors += torch.sum(torch.pow(y_pred - y, 2)).to(self._device)
        # Upcast y to float64 caller-side. WelfordVariance is dtype-agnostic
        # and the float64 cast is the only thing that buys back the stability
        # the original Σy² − (Σy)²/n formula was losing.
        self._y_running_stats.update(y.to(torch.float64).flatten())

    @sync_all_reduce("_num_examples", "_sum_of_errors")
    def compute(self) -> float:
        if self._num_examples == 0:
            raise NotComputableError("R2Score must have at least one example before it can be computed.")

        # Welford state is per-rank and not summable. The right reduction
        # is all_gather followed by pairwise merge using the Chan parallel
        # formula. See ignite.metrics._running_stats module docstring.
        ws = self._y_running_stats
        if idist.get_world_size() > 1:
            gathered = idist.all_gather(ws)
            ws = WelfordVariance()
            for item in gathered:
                ws.merge(item)

        denominator = ws.sum_sq_dev_from_mean.item()
        if denominator == 0.0:
            raise NotComputableError(
                "R2Score is undefined when the ground truth has zero variance (all y values are identical)."
            )

        return 1 - self._sum_of_errors.item() / denominator
