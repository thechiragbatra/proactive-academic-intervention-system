"""
OOP — NotificationEngine.

Responsibilities
----------------
- Decide WHO gets an alert (risk band + audit trigger logic).
- Decide WHAT the message says (templated per risk band + recommendation).
- Decide HOW to deliver it (console / JSON / SMTP).

The synopsis calls for SMTP support. We ship a `SMTPDispatcher` that is
opt-in and requires explicit credentials — by default, everything goes to
the `ConsoleDispatcher`, which is safe for demos and viva.

Inheritance
-----------
    Notification  ← base dataclass
    StudentNotification    (inherits)
    ParentNotification     (inherits)

The dispatchers share a common abstract interface `Dispatcher`, and
concrete implementations override `send()`.
"""
from __future__ import annotations
import json
import smtplib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Iterable

from .. import config as C
from .student_record import StudentRecord


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Notification:
    student_id: str
    to_email: str
    subject: str
    body: str
    risk_band: str
    recipient: str = "student"          # "student" | "parent"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict: return asdict(self)


def StudentNotification(**kwargs) -> Notification:
    """Factory — builds a Notification with recipient='student'."""
    kwargs.setdefault("recipient", "student")
    return Notification(**kwargs)


def ParentNotification(**kwargs) -> Notification:
    """Factory — builds a Notification with recipient='parent'."""
    kwargs["recipient"] = "parent"
    return Notification(**kwargs)


# ---------------------------------------------------------------------------
# Dispatchers
# ---------------------------------------------------------------------------
class Dispatcher(ABC):
    @abstractmethod
    def send(self, n: Notification) -> None: ...


class ConsoleDispatcher(Dispatcher):
    """Prints notifications — default for demos and tests."""
    def send(self, n: Notification) -> None:
        print(f"[{n.created_at}] → {n.recipient.upper()} <{n.to_email}>")
        print(f"  Subject: {n.subject}")
        print(f"  Body   : {n.body[:200]}{'…' if len(n.body) > 200 else ''}")


class JsonlDispatcher(Dispatcher):
    """Appends notifications to a JSON-lines file — useful for audit logs."""
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def send(self, n: Notification) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(n.to_dict(), ensure_ascii=False) + "\n")


class SMTPDispatcher(Dispatcher):
    """
    Sends over SMTP. DISABLED by default — callers must construct explicitly
    with host/port/user/password. Included to satisfy the synopsis's
    "SMTP Library" requirement.
    """
    def __init__(self, *, host: str, port: int,
                 username: str, password: str, from_addr: str) -> None:
        self.host, self.port = host, port
        self.username, self.password = username, password
        self.from_addr = from_addr

    def send(self, n: Notification) -> None:
        msg = EmailMessage()
        msg["From"] = self.from_addr
        msg["To"] = n.to_email
        msg["Subject"] = n.subject
        msg.set_content(n.body)
        with smtplib.SMTP_SSL(self.host, self.port) as s:
            s.login(self.username, self.password)
            s.send_message(msg)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class NotificationEngine:
    """Decides and dispatches alerts for a set of students."""

    SUBJECT_BY_BAND = {
        "CRITICAL": "URGENT: Immediate Academic Review Required",
        "HIGH":     "Important: Academic Performance Audit",
        "MODERATE": "Check-in: Keep Your Momentum",
    }

    def __init__(self, dispatcher: Dispatcher | None = None) -> None:
        self.dispatcher = dispatcher or ConsoleDispatcher()
        self.sent_log: list[Notification] = []

    # ------------------------------------------------------------------
    # Decision layer
    # ------------------------------------------------------------------
    @staticmethod
    def should_notify(record: StudentRecord,
                      marks_reflected_pct: float = 0.0) -> bool:
        """
        Synopsis "50-mark audit": once ≥50% of marks are in, we notify
        every student whose risk band is MODERATE or worse.
        """
        if record.risk_band in {"CRITICAL", "HIGH"}:
            return True
        if (record.risk_band == "MODERATE"
                and marks_reflected_pct >= C.AUDIT_TRIGGER_MARKS_PCT):
            return True
        return False

    # ------------------------------------------------------------------
    # Message construction
    # ------------------------------------------------------------------
    def _build_student_message(self, r: StudentRecord,
                               recommendation_text: str) -> Notification:
        subject = self.SUBJECT_BY_BAND.get(r.risk_band, "Academic Check-in")
        body = (
            f"Dear {r.first_name},\n\n"
            f"Your current academic profile suggests a "
            f"{r.risk_band.lower()} risk of missing your target grade. "
            f"This is an early warning — there is still time to course-correct.\n\n"
            f"Snapshot:\n"
            f"  • Attendance      : {r.attendance:.1f}%\n"
            f"  • Midterm         : {r.midterm:.1f} / 100\n"
            f"  • Assignments avg : {r.assignments_avg:.1f} / 100\n"
            f"  • Participation   : {r.participation:.1f} / 100\n"
            f"  • Risk score      : {r.risk_score:.2f}\n\n"
            f"Personalised plan:\n{recommendation_text}\n\n"
            f"Please schedule a 15-minute check-in with your mentor this week.\n\n"
            f"— Office of Academic Success"
        )
        return StudentNotification(
            student_id=r.student_id, to_email=r.email,
            subject=subject, body=body, risk_band=r.risk_band,
        )

    def _build_parent_message(self, r: StudentRecord) -> Notification:
        # We don't store guardian email; use a deterministic placeholder.
        parent_email = f"parent_of_{r.student_id.lower()}@university.com"
        subject = f"Academic Status Update for {r.full_name}"
        body = (
            f"Dear Parent / Guardian,\n\n"
            f"This is a scheduled academic status update from the University.\n\n"
            f"Student      : {r.full_name} ({r.student_id})\n"
            f"Department   : {r.department}\n"
            f"Current Risk : {r.risk_band}\n"
            f"Attendance   : {r.attendance:.1f}%\n"
            f"Midterm      : {r.midterm:.1f} / 100\n\n"
            f"Our early-warning system flagged this student for intervention. "
            f"A detailed recovery plan has been shared with them directly; "
            f"your encouragement at home materially improves outcomes.\n\n"
            f"— Office of Academic Success"
        )
        return ParentNotification(
            student_id=r.student_id, to_email=parent_email,
            subject=subject, body=body, risk_band=r.risk_band,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def notify_student(self, r: StudentRecord, recommendation_text: str) -> None:
        n = self._build_student_message(r, recommendation_text)
        self.dispatcher.send(n)
        self.sent_log.append(n)

    def notify_parent(self, r: StudentRecord) -> None:
        n = self._build_parent_message(r)
        self.dispatcher.send(n)
        self.sent_log.append(n)

    def batch_notify(self, cohort: Iterable[StudentRecord],
                     recommendations: dict[str, str],
                     *, notify_parents_for: set[str] | None = None,
                     marks_reflected_pct: float = 0.0) -> None:
        """
        Iterate a cohort, firing student notifications (and parent
        notifications for the bands in `notify_parents_for`).
        """
        notify_parents_for = notify_parents_for or {"CRITICAL"}
        for r in cohort:
            if not self.should_notify(r, marks_reflected_pct):
                continue
            self.notify_student(r, recommendations.get(r.student_id, ""))
            if r.risk_band in notify_parents_for:
                self.notify_parent(r)
