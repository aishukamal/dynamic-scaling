"""Small helpers shared by the verl-facing modules. Importable without verl/ray (lazy imports)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


def unwrap_actor_class(obj: Any) -> type:
    """Return the plain Python class behind a ``@ray.remote``-decorated class.

    At verl d4701e4e, ``FullyAsyncTrainer``, ``FullyAsyncRollouter`` and ``FullyAsyncTaskRunner``
    all carry the decorator on the class definition (the fork's trainer registry that moved it to
    the instantiation site is NOT in the runtime image), and Ray refuses ``class X(ActorClass)``.
    Ray stores the original class as ``ActorClass.__ray_metadata__.modified_class.__ray_actor_class__``.
    """
    meta = getattr(obj, "__ray_metadata__", None)
    if meta is None:
        return obj
    modified = getattr(meta, "modified_class", None)
    if modified is None:
        return obj
    return getattr(modified, "__ray_actor_class__", modified)


def head_node_id() -> str | None:
    """Node id of the Ray head (resource ``node:__internal_head__``), or None."""
    import ray

    for node in ray.nodes():
        if node.get("Alive") and "node:__internal_head__" in (node.get("Resources") or {}):
            return node["NodeID"]
    return None


@dataclass
class RLPoolConfig:
    """``rl_pool.*`` config keys (see config/rl_pool_fully_async.yaml)."""

    min_standalone: int = 1
    max_standalone: int = 8
    prestart: bool = True
    control_port: int = 8787
    control_actor_name: str = "rl_pool_control"
    release_on_scale_in: bool = False
    joiner_sync_mode: str = "clean_only"  # "clean_only" | "immediate"
    join_timeout_s: float = 3600.0
    join_poll_s: float = 5.0
    scale_timeout_s: float = 1800.0  # ray.get timeout for trainer/rollouter scale RPCs
    status_timeout_s: float = 30.0  # ray.get timeout for rollouter status RPCs

    def validate(self) -> None:
        if self.min_standalone < 1:
            raise ValueError("rl_pool.min_standalone must be >= 1 (an empty standalone pool breaks the LB client)")
        if self.max_standalone < self.min_standalone:
            raise ValueError("rl_pool.max_standalone must be >= rl_pool.min_standalone")
        if self.joiner_sync_mode not in ("clean_only", "immediate"):
            raise ValueError("rl_pool.joiner_sync_mode must be 'clean_only' or 'immediate'")


def rl_pool_config(config: Any) -> RLPoolConfig:
    """Read the ``rl_pool`` section from a Hydra/OmegaConf config or a plain dict (missing -> defaults)."""
    section: Any = None
    if isinstance(config, dict):
        section = config.get("rl_pool")
    else:
        try:
            from omegaconf import OmegaConf

            if OmegaConf.is_config(config):
                section = OmegaConf.select(config, "rl_pool", default=None)
                if section is not None:
                    section = OmegaConf.to_container(section, resolve=True)
            else:
                getter = getattr(config, "get", None)
                section = getter("rl_pool", None) if getter is not None else None
        except ImportError:
            getter = getattr(config, "get", None)
            section = getter("rl_pool", None) if getter is not None else None
    if section is None:
        return RLPoolConfig()
    known = {f.name for f in fields(RLPoolConfig)}
    unknown = set(section) - known
    if unknown:
        raise ValueError(f"unknown rl_pool keys: {sorted(unknown)}; known: {sorted(known)}")
    cfg = RLPoolConfig(**{k: v for k, v in section.items() if k in known})
    cfg.validate()
    return cfg
