"""
DSA #1 — Priority Queue (Max-Heap) for ranking at-risk students.

Problem
-------
Mentors can only intervene with a handful of students at a time. Given
thousands of students with evolving risk scores, we need O(1) access to the
most critical cases and O(log n) updates as new data streams in.

Implementation
--------------
Python's heapq is a min-heap, so we invert the score (store `-risk_score`)
to simulate a max-heap. Each entry carries a stable tiebreaker (`order`) to
guarantee deterministic ordering when two students share the same score.

Complexity
----------
- push / update: O(log n)
- peek top k  : O(k log n)
- build from list of n: O(n log n) via repeated push
"""
from __future__ import annotations
import heapq
import itertools
from dataclasses import dataclass, field
from typing import Iterable


@dataclass(order=True)
class _HeapEntry:
    """Internal heap entry. Orders by (neg_score, order)."""
    neg_score: float
    order: int
    student_id: str = field(compare=False)
    metadata: dict = field(default_factory=dict, compare=False)


class RiskHeap:
    """Max-heap keyed on risk score. Higher score = bubbled to the top."""

    def __init__(self) -> None:
        self._heap: list[_HeapEntry] = []
        self._counter = itertools.count()
        # index: student_id -> heap entry, so we can mark stale on update.
        self._index: dict[str, _HeapEntry] = {}
        self._REMOVED = "<removed>"

    def __len__(self) -> int:
        return len(self._index)

    def push(self, student_id: str, risk_score: float,
             metadata: dict | None = None) -> None:
        """Insert or update a student's score. O(log n)."""
        if student_id in self._index:
            # Mark previous entry stale; real removal happens lazily at pop.
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
        """
        Return the top-k at-risk students without mutating the heap.

        Runs over a *copy* to stay non-destructive; O(k log n).
        """
        snapshot = list(self._heap)
        out: list[tuple[str, float, dict]] = []
        while snapshot and len(out) < k:
            entry = heapq.heappop(snapshot)
            if entry.student_id != self._REMOVED:
                out.append((entry.student_id, -entry.neg_score, entry.metadata))
        return out

    def pop_top(self) -> tuple[str, float, dict] | None:
        """Remove and return the single highest-risk student."""
        while self._heap:
            entry = heapq.heappop(self._heap)
            if entry.student_id != self._REMOVED:
                self._index.pop(entry.student_id, None)
                return entry.student_id, -entry.neg_score, entry.metadata
        return None

    def get(self, student_id: str) -> float | None:
        """Current risk score for a student, or None if not tracked."""
        entry = self._index.get(student_id)
        return -entry.neg_score if entry else None
