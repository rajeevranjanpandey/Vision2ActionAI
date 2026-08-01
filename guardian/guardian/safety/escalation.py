"""Countdown-then-notify escalation for a confirmed fall.

The classifier is only half the feature. The other half is the social contract around
it: no automatic message goes out without giving the wearer a chance to cancel, and the
cancel window has to be spoken *and* haptic because a person on the ground may not be
able to find a button.

The state machine is intentionally small and fully synchronous so it can be unit-tested
against a clock rather than wall time:

    IDLE --fall--> COUNTDOWN --timeout--> NOTIFYING --ok--> NOTIFIED
                       |                      |
                    cancel                  fail
                       v                      v
                     IDLE                  FAILED

Transport is injected. In the wearable it is a cellular SMS modem; in tests it is a list.
No app is required on the contact's end for v1 -- that is the whole point of SMS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from ..config import FallConfig
from .fall import FallEvent


class EscalationState(str, Enum):
    IDLE = "idle"
    COUNTDOWN = "countdown"
    NOTIFYING = "notifying"
    NOTIFIED = "notified"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(slots=True)
class TrustedContact:
    """A single pre-registered contact. v1 is deliberately one person, not a roster."""

    name: str
    phone_e164: str

    def __post_init__(self) -> None:
        if not self.phone_e164.startswith("+") or not self.phone_e164[1:].isdigit():
            raise ValueError("phone_e164 must be E.164, e.g. +447700900123")


@dataclass(slots=True)
class ContactAlert:
    """The message handed to the transport."""

    contact: TrustedContact
    body: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    event: Optional[FallEvent] = None

    @property
    def maps_url(self) -> str | None:
        if self.latitude is None or self.longitude is None:
            return None
        return f"https://maps.google.com/?q={self.latitude:.5f},{self.longitude:.5f}"


Transport = Callable[[ContactAlert], bool]
Speaker = Callable[[str], None]


@dataclass(slots=True)
class GpsFix:
    latitude: float
    longitude: float
    accuracy_m: float = 15.0
    age_s: float = 0.0


class FallEscalation:
    """Drives countdown, cancellation, and dispatch. Never touches the fast path."""

    def __init__(
        self,
        contact: TrustedContact,
        transport: Transport,
        cfg: FallConfig | None = None,
        speaker: Speaker | None = None,
    ) -> None:
        self.cfg = cfg or FallConfig()
        self.contact = contact
        self.transport = transport
        self.speaker = speaker or (lambda _msg: None)
        self.state = EscalationState.IDLE
        self.event: FallEvent | None = None
        self.sent: list[ContactAlert] = []
        self._deadline: float = 0.0
        self._last_spoken_at: float = -1e9
        self._attempts: int = 0
        self.spoken: list[str] = []

    # ---------------------------------------------------------------- triggering

    def on_fall(self, event: FallEvent, now_s: float) -> None:
        """Begin the countdown. Re-triggering during a live countdown is ignored."""
        if self.state in (EscalationState.COUNTDOWN, EscalationState.NOTIFYING):
            return
        self.event = event
        self.state = EscalationState.COUNTDOWN
        self._deadline = now_s + self.cfg.countdown_s
        self._attempts = 0
        self._last_spoken_at = -1e9
        self._say(
            f"Possible fall detected. Contacting {self.contact.name} in "
            f"{int(self.cfg.countdown_s)} seconds. Say cancel to stop.",
            now_s,
        )

    def trigger_manual(self, now_s: float) -> None:
        """Wearer-initiated SOS. Same channel, shorter fuse, no classifier involved."""
        self.state = EscalationState.COUNTDOWN
        self.event = None
        self._deadline = now_s + min(self.cfg.countdown_s, 10.0)
        self._attempts = 0
        self._say(f"Sending an emergency message to {self.contact.name}.", now_s)

    def cancel(self, now_s: float) -> bool:
        """Wearer says "I'm fine". Returns True if a pending alert was actually stopped."""
        if self.state is not EscalationState.COUNTDOWN:
            return False
        self.state = EscalationState.CANCELLED
        self._say("Cancelled. No message sent.", now_s)
        return True

    # -------------------------------------------------------------------- clock

    def remaining_s(self, now_s: float) -> float:
        if self.state is not EscalationState.COUNTDOWN:
            return 0.0
        return max(0.0, self._deadline - now_s)

    def tick(self, now_s: float, fix: GpsFix | None = None) -> ContactAlert | None:
        """Advance the machine. Call from the async channel, not the fast path."""
        if self.state is not EscalationState.COUNTDOWN:
            return None

        remaining = self.remaining_s(now_s)
        if remaining > 0.0:
            if now_s - self._last_spoken_at >= self.cfg.countdown_prompt_s:
                self._say(f"{int(round(remaining))} seconds. Say cancel to stop.", now_s)
            return None

        self.state = EscalationState.NOTIFYING
        alert = self._compose(fix)
        for _ in range(max(1, self.cfg.max_send_attempts)):
            self._attempts += 1
            try:
                ok = bool(self.transport(alert))
            except Exception:  # a dead modem must not take the device down
                ok = False
            if ok:
                self.state = EscalationState.NOTIFIED
                self.sent.append(alert)
                self._say(f"{self.contact.name} has been notified.", now_s)
                return alert
        self.state = EscalationState.FAILED
        self._say(
            f"Could not reach {self.contact.name}. No signal. Trying again is up to you.",
            now_s,
        )
        return None

    @property
    def attempts(self) -> int:
        return self._attempts

    # ---------------------------------------------------------------- internals

    def _compose(self, fix: GpsFix | None) -> ContactAlert:
        who = "Guardian wearer"
        if self.event is not None:
            detail = f"Fall detected ({self.event.summary}). No response for " \
                     f"{int(self.cfg.countdown_s)}s."
        else:
            detail = "Emergency requested by the wearer."
        body = f"{who}: {detail}"
        if fix is not None:
            body += (
                f" Last known location: {fix.latitude:.5f},{fix.longitude:.5f}"
                f" (+/-{fix.accuracy_m:.0f} m, {fix.age_s:.0f}s old)."
            )
        else:
            body += " No GPS fix available."
        return ContactAlert(
            contact=self.contact,
            body=body,
            latitude=fix.latitude if fix else None,
            longitude=fix.longitude if fix else None,
            event=self.event,
        )

    def _say(self, message: str, now_s: float) -> None:
        self._last_spoken_at = now_s
        self.spoken.append(message)
        self.speaker(message)


@dataclass(slots=True)
class NullTransport:
    """Dry-run transport: records instead of sending. Default in dev and in tests."""

    outbox: list[ContactAlert] = field(default_factory=list)
    fail: bool = False

    def __call__(self, alert: ContactAlert) -> bool:
        if self.fail:
            return False
        self.outbox.append(alert)
        return True
