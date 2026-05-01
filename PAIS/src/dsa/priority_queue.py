from __future__ import annotations
import heapq
import itertools
from dataclasses import dataclass, field
from typing import Iterable


@dataclass(order=True)
class _HeapEntry:
    neg_score: float
    order: int
    student_id: str = field(compare=False)
    metadata: dict = field(default_factory=dict, compare=False)


class RiskHeap:

    def __init__(self) -> None:
        self._heap: list[_HeapEntry] = []
        self._counter = itertools.count()

        self._index: dict[str, _HeapEntry] = {}
        self._REMOVED = "<removed>"

    def __len__(self) -> int:
        return len(self._index)

    def push(self, student_id: str, risk_score: float,
             metadata: dict | None = None) -> None:
        if student_id in self._index:

            self._index[student_id].student_id = self._REMOVED

        entry = _HeapEntry(
            neg_score=-risk_score,
            order=next(self._counter),
            student_id=student_id,
            metadata=metadata or {},
        )
        self._index[student_id] = entry
        heapq.heappush(self._heap, entry)

    def bulk_load(self, rows: Iterable[tuple[str, float, dict | None]]) -> None:
        for sid, score, meta in rows:
            self.push(sid, score, meta)

    def peek_top(self, k: int = 1) -> list[tuple[str, float, dict]]:
        snapshot = list(self._heap)
        out: list[tuple[str, float, dict]] = []
        while snapshot and len(out) < k:
            entry = heapq.heappop(snapshot)
            if entry.student_id != self._REMOVED:
                out.append((entry.student_id, -entry.neg_score, entry.metadata))
        return out

    def pop_top(self) -> tuple[str, float, dict] | None:
        while self._heap:
            entry = heapq.heappop(self._heap)
            if entry.student_id != self._REMOVED:
                self._index.pop(entry.student_id, None)
                return entry.student_id, -entry.neg_score, entry.metadata
        return None

    def get(self, student_id: str) -> float | None:
        entry = self._index.get(student_id)
        return -entry.neg_score if entry else None
