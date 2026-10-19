"""Minimal Prometheus text exposition: render and parse. Pure Python, no client library."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

_LINE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>\S+)(?:\s+\d+)?$")
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


@dataclass
class Sample:
    name: str
    value: float
    labels: dict[str, str] = field(default_factory=dict)


def _fmt(v: Any) -> str:
    if v is None:
        return "NaN"
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    return repr(float(v))


def _labels(labels: dict[str, str] | None) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{k}="{str(v).replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"' for k, v in labels.items())
    return "{" + inner + "}"


def render(metrics: Iterable[tuple[str, Any, dict[str, str] | None, str | None, str | None]]) -> str:
    """Render samples. Each item: (name, value, labels, help, type). Groups HELP/TYPE per name."""
    out: list[str] = []
    seen: set[str] = set()
    for name, value, labels, help_, type_ in metrics:
        if name not in seen:
            seen.add(name)
            if help_:
                out.append(f"# HELP {name} {help_}")
            out.append(f"# TYPE {name} {type_ or 'gauge'}")
        out.append(f"{name}{_labels(labels)} {_fmt(value)}")
    return "\n".join(out) + "\n"


def parse(text: str) -> list[Sample]:
    samples: list[Sample] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        labels: dict[str, str] = {}
        if m.group("labels"):
            for k, v in _LABEL.findall(m.group("labels")):
                labels[k] = v.encode().decode("unicode_escape")
        raw_v = m.group("value")
        try:
            value = float("nan") if raw_v == "NaN" else float(raw_v)
        except ValueError:
            continue
        samples.append(Sample(name=m.group("name"), value=value, labels=labels))
    return samples


def scalar(samples: list[Sample], name: str, default: float | None = None) -> float | None:
    for s in samples:
        if s.name == name and not s.labels:
            return s.value
    for s in samples:
        if s.name == name:
            return s.value
    return default


def by_label(samples: list[Sample], name: str, label: str) -> dict[str, float]:
    return {s.labels[label]: s.value for s in samples if s.name == name and label in s.labels}
