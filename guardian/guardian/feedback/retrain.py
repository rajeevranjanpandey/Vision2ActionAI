"""The manual retrain cycle, and the metric that decides whether it was worth running.

The roadmap's kill-or-keep number for item 5 is *late alerts on held-out real-world
clips trending toward zero* -- the same bar the shipped head is already held to. A
federated round that improves mean loss while adding one late alert has failed.

The model here is a logistic head over the same flattened feature window as the deployed
ensemble. It is not the deployed architecture, and it does not pretend to be: the point
of this module is the *loop* -- clients hold their own flagged windows, compute a local
delta, the server clips-noises-averages, and the resulting model is scored on a held-out
set neither side trained on. Swapping in the real ensemble changes the client update
function and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .federated import ClientUpdate, aggregate


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _predict(w: np.ndarray, b: float, X: np.ndarray) -> np.ndarray:
    return _sigmoid(X @ w + b)


def local_update(
    w: np.ndarray, b: float, X: np.ndarray, y: np.ndarray,
    steps: int = 8, lr: float = 0.18, l2: float = 1e-4
) -> dict[str, np.ndarray]:
    """A few epochs of full-batch logistic descent on one client's flagged windows.

    Deliberately short. A client that trains to convergence on sixty of its own windows
    produces a delta that is mostly its own routes, and clipping then throws away the
    part that generalises along with the part that does not.
    """
    w_local, b_local = w.copy(), float(b)
    n = max(len(X), 1)
    for _ in range(steps):
        p = _predict(w_local, b_local, X)
        g = p - y
        w_local -= lr * ((X.T @ g) / n + l2 * w_local)
        b_local -= lr * float(g.mean())
    return {"w": w_local - w, "b": np.array([b_local - b])}


def late_alert_count(p_early: np.ndarray, p_late: np.ndarray, y: np.ndarray,
                     threshold: float) -> int:
    """Hazards the model warns about only once the evidence is already overwhelming.

    ``p_early`` is the score on an attenuated view of the same hazard -- the state a
    second earlier, when the closing signature is weaker. A late alert is a hazard the
    model *does* fire on, but only from the strong late view: the warning exists and
    arrives with no usable lead time.

    This is the number that separates a model that got better from a model that merely
    got louder. A recall column cannot tell the difference; on the pavement it is the
    whole difference.
    """
    fired_late = p_late >= threshold
    fired_early = p_early >= threshold
    return int(np.sum((y == 1) & fired_late & ~fired_early))


@dataclass(slots=True)
class RetrainRound:
    round_index: int
    holdout_recall: float
    holdout_false_alarm_rate: float
    late_alerts: int
    aggregate_norm: float
    clipped_fraction: float


def _make_population(rng: np.random.Generator, dim: int, n_clients: int,
                     per_client: int, n_holdout: int
                     ) -> tuple[list[tuple[np.ndarray, np.ndarray]],
                                tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Clients see a shifted slice of the world; the holdout is the pooled truth.

    The shift is the whole reason federation is interesting here: no single wearer's
    routes cover the distribution, so a model fitted on one device's log overfits that
    device's streets. The holdout is drawn without the shift.
    """
    true_w = rng.normal(0.0, 0.6, dim)
    clients = []
    for c in range(n_clients):
        shift = rng.normal(0.0, 0.5, dim) * (c / max(n_clients - 1, 1))
        X = rng.normal(0.0, 1.0, (per_client, dim)) + shift
        p = _sigmoid(X @ true_w)
        y = (rng.random(per_client) < p).astype(np.float64)
        clients.append((X, y))
    Xh = rng.normal(0.0, 1.0, (n_holdout, dim))
    ph = _sigmoid(Xh @ true_w)
    yh = (rng.random(n_holdout) < ph).astype(np.float64)
    # The "one second earlier" view of the same encounter: the same geometry with the
    # closing signature not yet developed. Lead time is measured against this.
    X_early = 0.70 * Xh
    return clients, (Xh, yh, X_early, true_w)


def simulate_retrain(
    rounds: int = 8,
    n_clients: int = 12,
    per_client: int = 60,
    dim: int = 24,
    n_holdout: int = 900,
    clip_norm: float = 1.0,
    noise_multiplier: float = 0.15,
    threshold: float = 0.60,
    seed: int = 0,
) -> dict[str, object]:
    """Run the loop and report the late-alert trend, not just the loss curve."""
    rng = np.random.default_rng(seed)
    clients, (Xh, yh, X_early, true_w) = _make_population(
        rng, dim, n_clients, per_client, n_holdout
    )

    # The round-zero model is the simulation-trained head as shipped: right about which
    # direction is dangerous, systematically under-confident about magnitude. That is the
    # realistic starting error, and it is exactly the error that produces late alerts --
    # the score crosses threshold, but only once the encounter is already close.
    w = 0.30 * true_w
    b = 0.0
    history: list[RetrainRound] = []

    for r in range(rounds):
        updates = [
            ClientUpdate(f"c{i:02d}", local_update(w, b, X, y), n_samples=len(X))
            for i, (X, y) in enumerate(clients)
        ]
        agg, report = aggregate(
            updates, clip_norm=clip_norm, noise_multiplier=noise_multiplier,
            rng=np.random.default_rng(seed + 1000 + r),
        )
        w = w + agg["w"]
        b = b + float(agg["b"][0])

        p = _predict(w, b, Xh)
        fired = p >= threshold
        recall = float(fired[yh == 1].mean()) if (yh == 1).any() else 0.0
        fa = float(fired[yh == 0].mean()) if (yh == 0).any() else 0.0
        history.append(
            RetrainRound(
                round_index=r + 1,
                holdout_recall=recall,
                holdout_false_alarm_rate=fa,
                late_alerts=late_alert_count(
                    _predict(w, b, X_early), p, yh, threshold
                ),
                aggregate_norm=report.aggregate_norm,
                clipped_fraction=report.clipped_fraction,
            )
        )

    return {
        "rounds": [
            {
                "round": h.round_index,
                "recall": round(h.holdout_recall, 4),
                "falseAlarmRate": round(h.holdout_false_alarm_rate, 4),
                "lateAlerts": h.late_alerts,
                "aggregateNorm": round(h.aggregate_norm, 4),
                "clippedFraction": round(h.clipped_fraction, 3),
            }
            for h in history
        ],
        "lateAlertsFirst": history[0].late_alerts if history else 0,
        "lateAlertsLast": history[-1].late_alerts if history else 0,
        "noiseMultiplier": noise_multiplier,
        "clipNorm": clip_norm,
        "nClients": n_clients,
    }


def privacy_utility_curve(
    noise_multipliers: tuple[float, ...] = (0.0, 0.2, 0.4, 0.8, 1.6),
    rounds: int = 6,
    seed: int = 0,
) -> list[dict[str, float]]:
    """What each privacy setting costs in held-out recall and late alerts."""
    out: list[dict[str, float]] = []
    for z in noise_multipliers:
        res = simulate_retrain(rounds=rounds, noise_multiplier=z, seed=seed)
        last = res["rounds"][-1]  # type: ignore[index]
        out.append(
            {
                "noiseMultiplier": z,
                "recall": last["recall"],
                "falseAlarmRate": last["falseAlarmRate"],
                "lateAlerts": last["lateAlerts"],
            }
        )
    return out
