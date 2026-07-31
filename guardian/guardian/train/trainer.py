"""Numpy trainer for the temporal risk head.

A 2-layer MLP over a flattened 0.8 s window, trained with focal loss and Adam, deep
ensembled 5x. No torch, no CUDA, ~2 seconds to fit on a laptop.

Justifying the architecture rather than reaching for a transformer:

* The input is 112 numbers (8 timesteps x 14 features). At that dimensionality the
  bottleneck is label quality, not capacity -- a GRU and an MLP-on-window land within
  noise of each other, and the MLP has a fixed 40 us inference cost that fits in a
  10 Hz budget already spending 90 ms on perception.
* Deep ensembles (Lakshminarayanan et al.) give an epistemic-uncertainty estimate for
  free. The policy uses that spread: high disagreement falls back to the geometric
  kernel instead of trusting a confident-looking single-model probability.
* Focal loss handles the ~9:1 negative skew without resampling, which would distort the
  base rate the calibrator later has to correct anyway.

The trained artefact is a plain ``.npz``. It is inspectable, diffable in review, and has
no pickle deserialisation surface -- which matters for a device you strap to a person.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrainConfig:
    hidden: tuple[int, int] = (64, 32)
    lr: float = 3e-3
    weight_decay: float = 1e-4
    epochs: int = 120
    batch_size: int = 256
    focal_gamma: float = 2.0
    focal_alpha: float = 0.6
    ensemble: int = 5
    patience: int = 15
    seed: int = 0


@dataclass
class TrainReport:
    val_loss: list[float] = field(default_factory=list)
    val_auc: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_auc: float = 0.0
    n_train: int = 0
    n_val: int = 0
    positive_rate: float = 0.0


# ------------------------------------------------------------------------- utilities


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic. The naive form overflows on the very confident
    negatives that dominate this dataset."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U). Ties get averaged ranks."""
    y = np.asarray(y).ravel()
    p = np.asarray(p).ravel()
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    sorted_p = p[order]
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y: np.ndarray, p: np.ndarray) -> float:
    """AP is the honest headline for a 9:1 skewed problem where AUC flatters."""
    y = np.asarray(y).ravel()
    p = np.asarray(p).ravel()
    if y.sum() == 0:
        return 0.0
    order = np.argsort(-p, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def group_split(groups: np.ndarray, val_fraction: float = 0.25,
                seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Split by clip id, never by frame. See ``features.build_windows``."""
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_fraction)))
    val_groups = set(uniq[:n_val].tolist())
    mask = np.array([g in val_groups for g in groups])
    return ~mask, mask


# ----------------------------------------------------------------------------- model


class MLP:
    """Two hidden layers, ReLU, single logit out. He init, Adam, decoupled weight decay."""

    def __init__(self, in_dim: int, hidden: tuple[int, int], seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        dims = [in_dim, hidden[0], hidden[1], 1]
        self.W = [rng.normal(0, np.sqrt(2.0 / dims[i]), (dims[i], dims[i + 1]))
                  for i in range(3)]
        self.b = [np.zeros(dims[i + 1]) for i in range(3)]
        self._m = [np.zeros_like(w) for w in self.W] + [np.zeros_like(v) for v in self.b]
        self._v = [np.zeros_like(w) for w in self.W] + [np.zeros_like(v) for v in self.b]
        self._t = 0

    # -- forward ---------------------------------------------------------------

    def logits(self, X: np.ndarray) -> np.ndarray:
        h1 = np.maximum(X @ self.W[0] + self.b[0], 0.0)
        h2 = np.maximum(h1 @ self.W[1] + self.b[1], 0.0)
        return (h2 @ self.W[2] + self.b[2]).ravel()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(self.logits(X))

    # -- training --------------------------------------------------------------

    def _forward_cache(self, X):
        z1 = X @ self.W[0] + self.b[0]
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ self.W[1] + self.b[1]
        h2 = np.maximum(z2, 0.0)
        z3 = (h2 @ self.W[2] + self.b[2]).ravel()
        return z1, h1, z2, h2, z3

    def step(self, X: np.ndarray, y: np.ndarray, cfg: TrainConfig) -> float:
        z1, h1, z2, h2, z3 = self._forward_cache(X)
        p = sigmoid(z3)
        eps = 1e-7
        pt = np.where(y == 1, p, 1.0 - p).clip(eps, 1.0)
        alpha_t = np.where(y == 1, cfg.focal_alpha, 1.0 - cfg.focal_alpha)
        g = cfg.focal_gamma
        loss = float(np.mean(-alpha_t * (1.0 - pt) ** g * np.log(pt)))

        # dL/dz for the focal loss, derived by hand rather than autodiffed:
        #   dL/dz = s * a_t * (1 - p_t)^g * (g * p_t * log p_t - (1 - p_t))
        # with s = +1 for positives and -1 for negatives. At g = 0 this collapses to
        # the plain BCE gradient (p - y), which is the check that it is right.
        sign = np.where(y == 1, 1.0, -1.0)
        dz = (sign * alpha_t * (1.0 - pt) ** g
              * (g * pt * np.log(pt) - (1.0 - pt))) / len(y)

        gW2 = h2.T @ dz[:, None]
        gb2 = dz.sum(keepdims=True)
        dh2 = np.outer(dz, self.W[2].ravel()) * (z2 > 0)
        gW1 = h1.T @ dh2
        gb1 = dh2.sum(axis=0)
        dh1 = (dh2 @ self.W[1].T) * (z1 > 0)
        gW0 = X.T @ dh1
        gb0 = dh1.sum(axis=0)

        self._adam([gW0, gW1, gW2, gb0, gb1, gb2], cfg)
        return loss

    def _adam(self, grads, cfg: TrainConfig, b1: float = 0.9, b2: float = 0.999):
        self._t += 1
        params = self.W + self.b
        for i, (param, grad) in enumerate(zip(params, grads)):
            self._m[i] = b1 * self._m[i] + (1 - b1) * grad
            self._v[i] = b2 * self._v[i] + (1 - b2) * grad**2
            mhat = self._m[i] / (1 - b1**self._t)
            vhat = self._v[i] / (1 - b2**self._t)
            update = cfg.lr * mhat / (np.sqrt(vhat) + 1e-8)
            if i < len(self.W):  # decoupled weight decay on weights only, not biases
                update += cfg.lr * cfg.weight_decay * param
            param -= update

    # -- serialisation ---------------------------------------------------------

    def state(self) -> dict[str, np.ndarray]:
        out = {f"W{i}": w.astype(np.float32) for i, w in enumerate(self.W)}
        out.update({f"b{i}": b.astype(np.float32) for i, b in enumerate(self.b)})
        return out

    @classmethod
    def from_state(cls, state: dict[str, np.ndarray]) -> "MLP":
        model = cls.__new__(cls)
        model.W = [np.asarray(state[f"W{i}"], dtype=np.float64) for i in range(3)]
        model.b = [np.asarray(state[f"b{i}"], dtype=np.float64) for i in range(3)]
        model._m, model._v, model._t = [], [], 0
        return model


# -------------------------------------------------------------------------- training


def train_ensemble(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                   cfg: TrainConfig | None = None) -> tuple[list[MLP], TrainReport, dict]:
    """Fit the deep ensemble and return the members, a report, and held-out scores.

    The held-out scores come back with the models because the calibration stage needs
    predictions the models never trained on -- calibrating on training predictions is
    the most common way a "well-calibrated" system ships badly overconfident.
    """
    cfg = cfg or TrainConfig()
    train_mask, val_mask = group_split(groups, seed=cfg.seed)
    Xtr, ytr = X[train_mask], y[train_mask]
    Xva, yva = X[val_mask], y[val_mask]

    report = TrainReport(n_train=int(len(ytr)), n_val=int(len(yva)),
                         positive_rate=float(y.mean()) if len(y) else 0.0)
    models: list[MLP] = []

    for member in range(cfg.ensemble):
        rng = np.random.default_rng(cfg.seed * 1000 + member)
        model = MLP(X.shape[1], cfg.hidden, seed=cfg.seed * 100 + member)
        # Bootstrap each member so disagreement reflects data uncertainty too, not just
        # init noise.
        idx_pool = rng.integers(0, len(ytr), len(ytr))
        best_auc, best_state, stale = 0.0, model.state(), 0

        for epoch in range(cfg.epochs):
            perm = rng.permutation(len(idx_pool))
            for start in range(0, len(perm), cfg.batch_size):
                batch = idx_pool[perm[start : start + cfg.batch_size]]
                if len(batch) < 8:
                    continue
                model.step(Xtr[batch], ytr[batch], cfg)

            if epoch % 3 == 0 or epoch == cfg.epochs - 1:
                auc = roc_auc(yva, model.predict(Xva)) if len(yva) else 0.5
                if member == 0:
                    report.val_auc.append(auc)
                if auc > best_auc + 1e-4:
                    best_auc, best_state, stale = auc, model.state(), 0
                    if member == 0:
                        report.best_epoch = epoch
                else:
                    stale += 1
                    if stale >= cfg.patience:
                        break

        models.append(MLP.from_state(best_state))
        report.best_auc = max(report.best_auc, best_auc)

    val_pred = np.mean([m.predict(Xva) for m in models], axis=0) if len(yva) else np.zeros(0)
    holdout = {"X": Xva, "y": yva, "p": val_pred, "mask": val_mask}
    return models, report, holdout


def save_ensemble(path: str, models: list[MLP], meta: dict) -> None:
    """Write ``.npz`` with every member plus the metadata the loader asserts against."""
    payload: dict[str, np.ndarray] = {}
    for i, model in enumerate(models):
        for k, v in model.state().items():
            payload[f"m{i}_{k}"] = v
    payload["n_members"] = np.array([len(models)])
    for k, v in meta.items():
        payload[f"meta_{k}"] = np.array(v)
    np.savez(path, **payload)


def load_ensemble(path: str) -> tuple[list[MLP], dict]:
    data = np.load(path, allow_pickle=False)
    n = int(data["n_members"][0])
    models = [
        MLP.from_state({key: data[f"m{i}_{key}"]
                        for key in ("W0", "W1", "W2", "b0", "b1", "b2")})
        for i in range(n)
    ]
    meta = {k[5:]: data[k] for k in data.files if k.startswith("meta_")}
    return models, meta
