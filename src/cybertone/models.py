"""Pydantic records persisted between pipeline stages."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class Trigger(BaseModel):
    """One sentence-level node in the source timeline."""

    tid: int
    src_in: float
    src_out: float
    text: str = ""
    n_words: int = 0

    @property
    def dur(self) -> float:
        return self.src_out - self.src_in


class PlanRow(BaseModel):
    tid: int
    src_in: float
    src_out: float
    dur: float
    text: str = ""
    r: float
    win_min: float
    win_max: float
    gap_before: float


class Plan(BaseModel):
    total: float
    target: float
    speech: float
    gap_budget: float
    b: float
    feasible: bool
    deficit: float = 0.0
    rows: list[PlanRow] = Field(default_factory=list)

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Plan:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class Stem(BaseModel):
    tid: int
    path: str
    dur: float
    speed: float
    win_min: float
    win_max: float
    degraded: bool = False
    qa: dict = Field(default_factory=dict)


def write_jsonl(path: Path, items: list[BaseModel]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(it.model_dump_json() + "\n")


def read_jsonl(path: Path, model: type[BaseModel]) -> list:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(model.model_validate(json.loads(line)))
    return out
