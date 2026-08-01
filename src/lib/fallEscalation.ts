/**
 * Browser port of `guardian/guardian/safety/escalation.py`.
 *
 * Same state machine, same countdown semantics, same message composition — so the
 * cancel window shown on this page is the one the wearable actually runs, not a mockup.
 */

export type EscalationState =
  | "idle"
  | "countdown"
  | "notifying"
  | "notified"
  | "cancelled"
  | "failed";

export interface TrustedContact {
  name: string;
  phoneE164: string;
}

export interface GpsFix {
  latitude: number;
  longitude: number;
  accuracyM: number;
  ageS: number;
}

export interface FallEvidence {
  peakG: number;
  freefallMs: number;
  tiltDeg: number;
  confidence: number;
  summary: string;
}

export interface EscalationConfig {
  countdownS: number;
  countdownPromptS: number;
}

export const DEFAULT_ESCALATION: EscalationConfig = {
  countdownS: 30,
  countdownPromptS: 5,
};

export interface EscalationSnapshot {
  state: EscalationState;
  remainingS: number;
  spoken: string[];
  message: string | null;
}

export class FallEscalation {
  private state: EscalationState = "idle";
  private deadline = 0;
  private lastSpokenAt = -1e9;
  private evidence: FallEvidence | null = null;
  private spoken: string[] = [];
  private message: string | null = null;

  constructor(
    private readonly contact: TrustedContact,
    private readonly cfg: EscalationConfig = DEFAULT_ESCALATION,
  ) {}

  reset(): void {
    this.state = "idle";
    this.spoken = [];
    this.message = null;
    this.evidence = null;
    this.lastSpokenAt = -1e9;
  }

  onFall(evidence: FallEvidence, nowS: number): void {
    if (this.state === "countdown" || this.state === "notifying") return;
    this.evidence = evidence;
    this.state = "countdown";
    this.deadline = nowS + this.cfg.countdownS;
    this.lastSpokenAt = -1e9;
    this.say(
      `Possible fall detected. Contacting ${this.contact.name} in ${Math.round(
        this.cfg.countdownS,
      )} seconds. Say cancel to stop.`,
      nowS,
    );
  }

  cancel(nowS: number): boolean {
    if (this.state !== "countdown") return false;
    this.state = "cancelled";
    this.say("Cancelled. No message sent.", nowS);
    return true;
  }

  tick(nowS: number, fix: GpsFix | null): void {
    if (this.state !== "countdown") return;
    const remaining = Math.max(0, this.deadline - nowS);
    if (remaining > 0) {
      if (nowS - this.lastSpokenAt >= this.cfg.countdownPromptS) {
        this.say(`${Math.round(remaining)} seconds. Say cancel to stop.`, nowS);
      }
      return;
    }
    this.state = "notifying";
    this.message = this.compose(fix);
    this.state = "notified";
    this.say(`${this.contact.name} has been notified.`, nowS);
  }

  snapshot(nowS: number): EscalationSnapshot {
    return {
      state: this.state,
      remainingS: this.state === "countdown" ? Math.max(0, this.deadline - nowS) : 0,
      spoken: [...this.spoken],
      message: this.message,
    };
  }

  private compose(fix: GpsFix | null): string {
    const detail = this.evidence
      ? `Fall detected (${this.evidence.summary}). No response for ${Math.round(
          this.cfg.countdownS,
        )}s.`
      : "Emergency requested by the wearer.";
    const where = fix
      ? ` Last known location: ${fix.latitude.toFixed(5)},${fix.longitude.toFixed(5)} (±${Math.round(
          fix.accuracyM,
        )} m, ${Math.round(fix.ageS)}s old).`
      : " No GPS fix available.";
    return `Guardian wearer: ${detail}${where}`;
  }

  private say(message: string, nowS: number): void {
    this.lastSpokenAt = nowS;
    this.spoken = [...this.spoken.slice(-5), message];
  }
}
