"""rl_pool_autoscaler: out-of-tree verl plugin + controller for an elastic STANDALONE sampler pool.

Layout (what imports what):

- ``membership``   pure Python. Trainer-side membership registry: the asyncio lock shared by the
                   regular weight sync and attach/detach, per-replica version bookkeeping, and the
                   locked attach / detach / update_weights / joiner-sync flows (duck-typed manager).
- ``orchestrator`` pure Python. ``ScaleOrchestrator`` converges the active standalone count to a
                   desired value one replica at a time, enforcing the ordering rules
                   (scale-out: wake|create -> attach -> joiner sync -> version gate -> LB add;
                   scale-in: LB remove -> abort -> detach -> sleep|destroy) against a ``PoolBackend``.
- ``prom``         pure Python. Prometheus text rendering and parsing.
- ``http_api``     pure Python. stdlib HTTP handler for GET /metrics, GET /status, POST /scale.
- ``controller``   pure Python. Kubernetes-side controller (policy + poll loop + JSONL decisions).
- ``rollouter``    needs verl. ``ElasticRollouter`` (FullyAsyncRollouter subclass): standalone
                   replica pool operations (create / park / wake / LB add+remove / destroy).
- ``trainer``      needs verl. ``ElasticTrainer`` (FullyAsyncTrainer subclass): attach_replicas /
                   detach_replicas / sync_joiners_now under the membership lock + metrics tap.
- ``control_actor`` needs ray. Named actor ``rl_pool_control`` hosting the HTTP API and reconciler.
- ``launcher``     needs verl. Drop-in replacement for fully_async_main.

Nothing in this package modifies verl source; everything is subclass / composition.
"""

__version__ = "0.1.0"
