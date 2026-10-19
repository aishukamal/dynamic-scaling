"""MetricsTap: proxy around verl's ``Tracking`` logger that keeps the latest logged numbers.

Pure Python (no verl import) so the key mapping can be tested locally. The trainer subclass does
``self.logger = MetricsTap(self.logger, on_log=...)``; stock code keeps calling
``self.logger.log(data=..., step=...)`` and every call is forwarded unchanged.

Keys read (what FullyAsyncTrainer logs at verl d4701e4e):
- from ``_fit_log_aggregated_training_metrics`` (MetricsAggregator output, once per sync cycle):
  ``fully_async/total_wait_time`` (SUM over the cycle's collections because the key contains
  "total"), ``timing_s/step`` (time_sum), ``timing_s/timing_s/param_sync`` (compute_timing_metrics
  prefixes the already-prefixed key), ``dynamic_resource/mq_size`` (last), and the counters.
- from ``_fit_update_weights`` (rollouter.reset_staleness timing): ``fully_async/rollouter/idle_ratio``
  (share of the version window the rollouter spent paused), ``fully_async/rollouter/version_time``.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


class MetricsTap:
    """Proxy around ``Tracking`` that keeps the latest numeric values of every logged dict."""

    _KEYS = {
        "trainer_wait_s": ("fully_async/total_wait_time",),
        "trainer_step_s": ("timing_s/step",),
        "last_sync_wall_s": ("timing_s/timing_s/param_sync", "timing_s/param_sync"),
        "mq_size": ("dynamic_resource/mq_size",),
        "rollouter_idle_ratio": ("fully_async/rollouter/idle_ratio",),
        "rollouter_version_time_s": ("fully_async/rollouter/version_time",),
        "stale_trajectories": ("fully_async/count/stale_trajectory_processed",),
        "param_version_logged": ("fully_async/count/current_param_version",),
        # time-weighted active_tasks / max_concurrent_samples over the version window (rollouter-side,
        # pool-size aware because _update_max_concurrent_samples records capacity changes)
        "rollout_util": ("dynamic_resource/rollout_resource_utilization",),
        "response_length_mean": ("response_length/mean",),
        "num_turns_mean": ("num_turns/mean",),
        "score_mean": ("critic/score/mean",),
        "dropped_stale": ("fully_async/count/dropped_stale_samples",),
        "aborted_ratio": ("response/aborted_ratio",),
        "partial_ratio": ("fully_async/partial/partial_ratio",),
    }

    def __init__(self, inner: Any, on_log: Callable[[dict[str, Any]], None] | None = None) -> None:
        self._inner = inner
        self._on_log = on_log
        self.latest: dict[str, float] = {}
        self.latest_step: int | None = None
        # step of the last AGGREGATED TRAINING log (the dict that carries timing_s/step). The rollouter's
        # reset_staleness dict and the version bump land a few ms before it, so a poller keyed on
        # current_param_version can read a new version with the previous version's step/wait (seen in run
        # m2-s1 at version 10). Controllers key "new sample" on this instead.
        self.train_log_step: int | None = None
        self.log_calls = 0
        self.last_log_t: float | None = None

    def log(self, data, step, backend=None):
        try:
            self._record(data, step)
        except Exception as e:  # noqa: BLE001 - never break training over telemetry
            logger.warning("MetricsTap record failed: %s", e)
        return self._inner.log(data=data, step=step, backend=backend)

    def __getattr__(self, name: str) -> Any:  # finish(), etc.
        return getattr(self._inner, name)

    def _record(self, data: dict, step: int) -> None:
        if not isinstance(data, dict):
            return
        for k, v in data.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            self.latest[k] = fv
        self.latest_step = int(step) if step is not None else self.latest_step
        if "timing_s/step" in data and step is not None:
            self.train_log_step = int(step)
        self.log_calls += 1
        self.last_log_t = time.time()
        if self._on_log is not None:
            self._on_log(self.control_snapshot())

    def control_snapshot(self) -> dict[str, Any]:
        snap: dict[str, Any] = {"log_step": self.latest_step, "train_log_step": self.train_log_step, "log_calls": self.log_calls, "last_log_t": self.last_log_t}
        for out, candidates in self._KEYS.items():
            for c in candidates:
                if c in self.latest:
                    snap[out] = self.latest[c]
                    break
        wait = snap.get("trainer_wait_s")
        step_s = snap.get("trainer_step_s")
        if wait is not None and step_s:
            snap["trainer_idle_ratio"] = max(0.0, min(1.0, wait / step_s))
        return snap
