"""
DSA #4 — Graph Theory (Bipartite graph + BFS) for engagement mapping.

Problem
-------
Disengagement is multi-dimensional. A student might attend classes but never
touch course materials. We model the access pattern as a bipartite graph:

    Students ─── edges ─── Resources

Edges carry weight = number of accesses. We then run BFS from each resource
node to answer: "which students are reachable / not reachable from the
core resource set?". Students with no edges (or only edges to peripheral
resources) are flagged as isolated.

This is a genuine graph-theoretic framing — sliding a threshold on edge
weight would give the same answer with a loop, but the graph structure
makes the intent explicit and opens the door to future work (e.g.,
recommending resources via collaborative filtering on the same graph).
"""
from __future__ import annotations
import pandas as pd
from collections import defaultdict, deque


class ResourceGraph:
    """
    Bipartite adjacency-list graph.

    Node types are distinguished by a prefix:
        "S:<id>"  for students
        "R:<id>"  for resources
    """

    def __init__(self) -> None:
        self.adj: dict[str, dict[str, int]] = defaultdict(dict)
        self.students: set[str] = set()
        self.resources: set[str] = set()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def add_edge(self, student_id: str, resource_id: str, weight: int = 1) -> None:
        s = f"S:{student_id}"
        r = f"R:{resource_id}"
        self.adj[s][r] = self.adj[s].get(r, 0) + weight
        self.adj[r][s] = self.adj[r].get(s, 0) + weight
        self.students.add(s)
        self.resources.add(r)

    def build_from_logs(self, logs: pd.DataFrame,
                        *, n_resources: int = 12) -> "ResourceGraph":
        """
        Build the graph from daily logs.

        We bucket each student's resource hits across `n_resources`
        synthetic resources by modding `day` — it simulates "different
        lectures / uploaded PDFs" getting access traffic across the term.
        """
        # Skip days with no access to keep the graph sparse.
        active = logs[logs["resource_hits"] > 0]
        for _, row in active.iterrows():
            sid = row["Student_ID"]
            rid = f"RES_{int(row['day']) % n_resources:02d}"
            self.add_edge(sid, rid, weight=int(row["resource_hits"]))
        return self

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def isolated_students(self, min_edges: int = 2) -> list[str]:
        """
        Students with fewer than `min_edges` distinct resource connections.

        Returns raw student IDs (prefix stripped).
        """
        isolated = []
        for s in self.students:
            if len(self.adj[s]) < min_edges:
                isolated.append(s[2:])   # strip "S:" prefix
        return isolated

    def bfs_reachable_students(self, from_resources: list[str]) -> set[str]:
        """
        BFS from a seed of core resources; return student IDs reached
        in 1 hop (one edge = one access event).

        Useful question: "Of the 5 most important lectures, which students
        have engaged with at least one?"
        """
        seen: set[str] = set()
        queue: deque = deque(f"R:{r}" for r in from_resources if f"R:{r}" in self.adj)
        while queue:
            node = queue.popleft()
            for neighbour in self.adj[node]:
                if neighbour.startswith("S:") and neighbour not in seen:
                    seen.add(neighbour)
        return {s[2:] for s in seen}

    def engagement_score(self, student_id: str) -> float:
        """
        Weighted degree centrality for a student, normalised to [0, 1].
        """
        s = f"S:{student_id}"
        if s not in self.adj:
            return 0.0
        raw = sum(self.adj[s].values())
        # Soft normaliser — 50 hits or more = fully engaged.
        return min(1.0, raw / 50.0)

    def summary(self) -> dict:
        return {
            "num_students": len(self.students),
            "num_resources": len(self.resources),
            "num_edges": sum(len(v) for k, v in self.adj.items()
                              if k.startswith("S:")),
            "isolated_count": len(self.isolated_students()),
        }
