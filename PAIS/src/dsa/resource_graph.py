from __future__ import annotations
import pandas as pd
from collections import defaultdict, deque


class ResourceGraph:

    def __init__(self) -> None:
        self.adj: dict[str, dict[str, int]] = defaultdict(dict)
        self.students: set[str] = set()
        self.resources: set[str] = set()


    def add_edge(self, student_id: str, resource_id: str, weight: int = 1) -> None:
        s = f"S:{student_id}"
        r = f"R:{resource_id}"
        self.adj[s][r] = self.adj[s].get(r, 0) + weight
        self.adj[r][s] = self.adj[r].get(s, 0) + weight
        self.students.add(s)
        self.resources.add(r)

    def build_from_logs(self, logs: pd.DataFrame,
                        *, n_resources: int = 12) -> "ResourceGraph":

        active = logs[logs["resource_hits"] > 0]
        for _, row in active.iterrows():
            sid = row["Student_ID"]
            rid = f"RES_{int(row['day']) % n_resources:02d}"
            self.add_edge(sid, rid, weight=int(row["resource_hits"]))
        return self


    def isolated_students(self, min_edges: int = 2) -> list[str]:
        isolated = []
        for s in self.students:
            if len(self.adj[s]) < min_edges:
                isolated.append(s[2:])
        return isolated

    def bfs_reachable_students(self, from_resources: list[str]) -> set[str]:
        seen: set[str] = set()
        queue: deque = deque(f"R:{r}" for r in from_resources if f"R:{r}" in self.adj)
        while queue:
            node = queue.popleft()
            for neighbour in self.adj[node]:
                if neighbour.startswith("S:") and neighbour not in seen:
                    seen.add(neighbour)
        return {s[2:] for s in seen}

    def engagement_score(self, student_id: str) -> float:
        s = f"S:{student_id}"
        if s not in self.adj:
            return 0.0
        raw = sum(self.adj[s].values())

        return min(1.0, raw / 50.0)

    def summary(self) -> dict:
        return {
            "num_students": len(self.students),
            "num_resources": len(self.resources),
            "num_edges": sum(len(v) for k, v in self.adj.items()
                              if k.startswith("S:")),
            "isolated_count": len(self.isolated_students()),
        }
