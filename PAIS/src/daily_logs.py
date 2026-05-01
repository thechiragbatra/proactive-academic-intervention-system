from __future__ import annotations
import numpy as np
import pandas as pd

from . import config as C


def generate_daily_logs(students: pd.DataFrame, *,
                        seed: int = C.RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []

    for _, row in students.iterrows():
        sid = row["Student_ID"]
        att_rate = row["Attendance (%)"] / 100.0
        hours_per_week = row["Study_Hours_per_Week"]
        lam = max(0.1, hours_per_week / 7.0)


        has_tail = rng.random() < 0.15
        tail_start = C.SIMULATED_TERM_DAYS - 21

        for day in range(1, C.SIMULATED_TERM_DAYS + 1):
            p_attend = att_rate
            hit_mult = 1.0
            if has_tail and day >= tail_start:

                decay = (day - tail_start) / 21.0
                p_attend *= max(0.1, 1.0 - decay)
                hit_mult *= max(0.1, 1.0 - decay)

            attended = int(rng.random() < p_attend)
            hits = int(rng.poisson(lam * hit_mult))
            records.append((sid, day, attended, hits))

    logs = pd.DataFrame.from_records(
        records, columns=["Student_ID", "day", "attended", "resource_hits"]
    )
    return logs


def persist(logs: pd.DataFrame) -> None:
    C.ensure_dirs()
    logs.to_csv(C.DATA_DAILY_LOGS, index=False)


if __name__ == "__main__":
    from .preprocessing import load_raw
    df = load_raw()
    print(f"Generating logs for {len(df)} students "
          f"× {C.SIMULATED_TERM_DAYS} days...")
    logs = generate_daily_logs(df)
    persist(logs)
    print(f"Wrote {len(logs):,} rows → {C.DATA_DAILY_LOGS}")
