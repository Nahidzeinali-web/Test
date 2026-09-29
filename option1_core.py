"""
Shared code for the Option 1 thesis experiments: robot failure detection from motion prediction.

Used by option1_final_experiments.ipynb. One `Experiment` = one random split of the recordings into
train / validation / test + training of all methods + evaluation on injected failures.
"""
import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import IsolationForest

ALL_NAMES = [f"joint{i}" for i in range(6)] + ["x", "y", "z", "q0", "q1", "q2", "q3", "gripper", "unused"]
FAILURES = ["jerk", "freeze", "drift", "noise"]
FAIL_LEN = {"jerk": 5, "freeze": 15, "drift": 20, "noise": 10}
METHODS = ["Rule: stays still", "Rule: keeps its speed", "Isolation Forest", "LSTM autoencoder",
           "GRU forecaster", "Transformer forecaster", "Uncertainty-aware GRU"]
FAMILY = {"Rule: stays still": "rule", "Rule: keeps its speed": "rule", "Isolation Forest": "classic",
          "LSTM autoencoder": "rebuild", "GRU forecaster": "predict", "Transformer forecaster": "predict",
          "Uncertainty-aware GRU": "new idea"}


# ----------------------------------------------------------------------------- data
def fix_quaternion_signs(s, q=slice(9, 13)):
    """q and -q are the same orientation; undo sign flips so they don't look like jumps."""
    s = s.copy()
    for t in range(1, len(s)):
        if np.dot(s[t, q], s[t - 1, q]) < 0:
            s[t, q] *= -1
    return s


def load_recordings(path):
    """Returns a dict with the cleaned recordings and their metadata."""
    data = np.load(path)
    lengths = data["lengths"]
    off = np.concatenate([[0], np.cumsum(lengths)])
    raw = [data["states"][off[i]:off[i + 1]] for i in range(len(lengths))]
    flips = sum(int((np.einsum("ij,ij->i", s[1:, 9:13], s[:-1, 9:13]) < 0).sum()) for s in raw)
    states = [fix_quaternion_signs(s) for s in raw]
    keep = np.concatenate(states).std(0) > 1e-6
    names = [n for n, k in zip(ALL_NAMES, keep) if k]
    return {"states": [s[:, keep] for s in states], "names": names, "flips": flips,
            "instructions": data["instructions"], "file_names": data["file_names"]}


# ----------------------------------------------------------------------------- models
def with_velocity(x):
    return torch.cat([x, torch.diff(x, dim=1, prepend=x[:, :1])], dim=-1)


class Seq2SeqGRU(nn.Module):
    """Encoder reads the past K steps, decoder rolls H steps forward.
    With uncertainty=True it predicts a mean AND a variance for every future value."""

    def __init__(self, D, H, uncertainty=False, hidden=128):
        super().__init__()
        self.H, self.uncertainty = H, uncertainty
        self.encoder = nn.GRU(2 * D, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.decoder = nn.GRU(1, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, D * (2 if uncertainty else 1))

    def forward(self, x):
        _, h = self.encoder(with_velocity(x))
        out, _ = self.decoder(torch.zeros(len(x), self.H, 1, device=x.device), h)
        o = self.head(out)
        if self.uncertainty:
            mean, log_var = o.chunk(2, dim=-1)
            return mean, log_var.clamp(-10, 6)
        return o


class TransformerForecaster(nn.Module):
    def __init__(self, D, K, H, d_model=64, heads=4, layers=2):
        super().__init__()
        self.D, self.H = D, H
        self.inp = nn.Linear(2 * D, d_model)
        self.pos = nn.Parameter(torch.zeros(1, K, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, H * D)

    def forward(self, x):
        h = self.encoder(self.inp(with_velocity(x)) + self.pos)
        return self.head(h[:, -1]).view(-1, self.H, self.D)


class LSTMAutoencoder(nn.Module):
    def __init__(self, D, hidden=64, bottleneck=16):
        super().__init__()
        self.encoder = nn.LSTM(D, hidden, batch_first=True)
        self.squeeze = nn.Linear(hidden, bottleneck)
        self.expand = nn.Linear(bottleneck, hidden)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, D)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        rep = self.expand(self.squeeze(h[-1]))[:, None].repeat(1, x.shape[1], 1)
        y, _ = self.decoder(rep)
        return self.out(y)


def beta_nll_loss(pred, y, beta=0.5):
    """Gaussian negative log-likelihood, weighted by variance^beta for stable training (Seitzer et al., 2022)."""
    mean, log_var = pred
    var = log_var.exp()
    nll = 0.5 * (log_var + (y - mean) ** 2 / var)
    return (nll * var.detach() ** beta).mean()


def center(x):
    return x - x.mean(axis=1, keepdims=True)


def smooth(x):
    return pd.Series(x).rolling(3, min_periods=1).mean().to_numpy()


def roc_auc(scores, labels):
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def false_alarm_rate_at_recall(scores, labels, recall=0.9):
    """Share of normal steps above the threshold that catches `recall` of the failure steps."""
    thr = np.quantile(scores[labels], 1 - recall)
    return float(np.mean(scores[~labels] >= thr))


# ----------------------------------------------------------------------------- experiment
class Experiment:
    def __init__(self, rec, seed=0, K=10, H=10, device=None, verbose=True):
        self.rec, self.seed, self.K, self.H = rec, seed, K, H
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        names = rec["names"]
        self.D = len(names)
        self.GRIPPER = names.index("gripper")
        self.XYZ = [names.index(n) for n in ("x", "y", "z")]
        self.CONT = [i for i in range(self.D) if i != self.GRIPPER]

        n = len(rec["states"])
        order = np.random.default_rng(seed).permutation(n)
        n_val = n_test = int(0.15 * n)
        self.VAL, self.TEST, self.TRAIN = order[:n_val], order[n_val:n_val + n_test], order[n_val + n_test:]

        tr = np.concatenate([rec["states"][i] for i in self.TRAIN])
        self.MU, self.SD = tr.mean(0), tr.std(0)
        self.Z = [((s - self.MU) / self.SD).astype(np.float32) for s in rec["states"]]
        self.STEP_SD = np.concatenate([np.diff(self.Z[i], axis=0) for i in self.TRAIN]).std(0)

        self.Xtr, self.Ytr = self._windows(self.TRAIN)
        self.Xva, self.Yva = self._windows(self.VAL)
        self.Xte, self.Yte = self._windows(self.TEST)
        self.Y_SD = self.Ytr.reshape(len(self.Ytr), -1).std(0).reshape(H, self.D) + 1e-6
        self.Y_SD_T = torch.tensor(self.Y_SD, device=self.device)
        self.train_time, self.history = {}, {}

    def log(self, msg):
        if self.verbose:
            print(msg)

    def _windows(self, idx):
        K, H = self.K, self.H
        X, Y = [], []
        for i in idx:
            z = self.Z[i]
            if len(z) < K + H:
                continue
            w = sliding_window_view(z, K + H, axis=0).transpose(0, 2, 1)
            X.append(w[:, :K]); Y.append(w[:, K:] - w[:, K - 1:K])
        return np.concatenate(X).astype(np.float32), np.concatenate(Y).astype(np.float32)

    # ---------------------------------------------------------------- training
    def _train_torch(self, model, loss_fn, train, val, epochs=80, batch=512, lr=2e-3, patience=10):
        t0 = time.time()
        model = model.to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        train = [torch.as_tensor(t, device=self.device) for t in train]
        val = [torch.as_tensor(t, device=self.device) for t in val]
        best, best_state, wait, hist = np.inf, None, 0, []
        for _ in range(epochs):
            model.train()
            perm = torch.randperm(len(train[0]), device=self.device)
            losses = []
            for i in range(0, len(perm), batch):
                b = perm[i:i + batch]
                loss = loss_fn(model, *[t[b] for t in train])
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); losses.append(loss.item())
            model.eval()
            with torch.no_grad():
                v = float(np.mean([loss_fn(model, *[t[i:i + 4096] for t in val]).item()
                                   for i in range(0, len(val[0]), 4096)]))
            hist.append((float(np.mean(losses)), v))
            if v < best - 1e-5:
                best, best_state, wait = v, copy.deepcopy(model.state_dict()), 0
            else:
                wait += 1
                if wait >= patience:
                    break
        model.load_state_dict(best_state)
        model.eval()
        return model, hist, time.time() - t0

    def _seed(self, offset=0):
        np.random.seed(self.seed + offset)
        torch.manual_seed(self.seed + offset)

    def train_all(self):
        ysd = self.Y_SD_T
        mse = lambda m, x, y: F.mse_loss(m(x), y / ysd)
        nll = lambda m, x, y: beta_nll_loss(m(x), y / ysd)
        data = ((self.Xtr, self.Ytr), (self.Xva, self.Yva))

        self._seed(1)
        self.gru, self.history["GRU forecaster"], self.train_time["GRU forecaster"] = \
            self._train_torch(Seq2SeqGRU(self.D, self.H), mse, *data)
        self._seed(2)
        self.ugru, self.history["Uncertainty-aware GRU"], self.train_time["Uncertainty-aware GRU"] = \
            self._train_torch(Seq2SeqGRU(self.D, self.H, uncertainty=True), nll, *data, lr=1e-3)
        self._seed(3)
        self.transformer, self.history["Transformer forecaster"], self.train_time["Transformer forecaster"] = \
            self._train_torch(TransformerForecaster(self.D, self.K, self.H), mse, *data, lr=1e-3)

        self.AE_SCALE = np.maximum(center(self.Xtr).reshape(-1, self.D).std(0), 1e-3).astype(np.float32)
        self._seed(4)
        self.ae, self.history["LSTM autoencoder"], self.train_time["LSTM autoencoder"] = self._train_torch(
            LSTMAutoencoder(self.D), lambda m, x: F.mse_loss(m(x), x),
            (self._ae_input(self.Xtr),), (self._ae_input(self.Xva),))
        self.AE_RES_SD = np.maximum(
            (self._ae_reconstruct(self._ae_input(self.Xva)) - self._ae_input(self.Xva)).reshape(-1, self.D).std(0), 1e-3)

        t0 = time.time()
        sample = np.random.default_rng(self.seed).choice(len(self.Xtr), size=min(20000, len(self.Xtr)), replace=False)
        self.iforest = IsolationForest(n_estimators=200, random_state=self.seed, n_jobs=-1).fit(
            self._if_features(self.Xtr[sample]))
        self.train_time["Isolation Forest"] = time.time() - t0

        self.RES_SD = {name: np.maximum((self.predict(name, self.Xva)[0] - self.Yva).std(0), 1e-3)
                       for name in ("Rule: stays still", "Rule: keeps its speed", "GRU forecaster",
                                    "Transformer forecaster", "Uncertainty-aware GRU")}
        self.log("  trained: " + ", ".join(f"{k} {v:.0f}s" for k, v in self.train_time.items()))

    # ---------------------------------------------------------------- predictions
    def _net(self, model, X, uncertainty=False):
        means, sds = [], []
        with torch.no_grad():
            for i in range(0, len(X), 4096):
                x = torch.as_tensor(np.ascontiguousarray(X[i:i + 4096]), device=self.device)
                if uncertainty:
                    m, lv = model(x)
                    means.append((m * self.Y_SD_T).cpu().numpy())
                    sds.append(((0.5 * lv).exp() * self.Y_SD_T).cpu().numpy())
                else:
                    means.append((model(x) * self.Y_SD_T).cpu().numpy())
        if not means:
            return np.zeros((0, self.H, self.D), np.float32), None
        return np.concatenate(means), (np.concatenate(sds) if uncertainty else None)

    def predict(self, name, X):
        """Future changes (N,H,D) and, for the uncertainty model, predicted spread (N,H,D)."""
        if name == "Rule: stays still":
            return np.zeros((len(X), self.H, self.D), np.float32), None
        if name == "Rule: keeps its speed":
            v = X[:, -1] - X[:, -2]
            return (np.arange(1, self.H + 1)[None, :, None] * v[:, None, :]).astype(np.float32), None
        if name == "GRU forecaster":
            return self._net(self.gru, X)
        if name == "Transformer forecaster":
            return self._net(self.transformer, X)
        if name == "Uncertainty-aware GRU":
            return self._net(self.ugru, X, uncertainty=True)
        raise KeyError(name)

    def _ae_input(self, X):
        return (center(X) / self.AE_SCALE).astype(np.float32)

    def _ae_reconstruct(self, Xin):
        out = []
        with torch.no_grad():
            for i in range(0, len(Xin), 4096):
                out.append(self.ae(torch.as_tensor(Xin[i:i + 4096], device=self.device)).cpu().numpy())
        return np.concatenate(out)

    def _if_features(self, X):
        return np.concatenate([self._ae_input(X).reshape(len(X), -1),
                               (np.diff(X, axis=1) / self.STEP_SD).reshape(len(X), -1)], axis=1)

    # ---------------------------------------------------------------- scores
    def score(self, name, z):
        """Unusualness score per time step (NaN where undefined); only uses the past."""
        K, H, T = self.K, self.H, len(z)
        if name == "LSTM autoencoder":
            W = sliding_window_view(z, K, axis=0).transpose(0, 2, 1)
            x = self._ae_input(W)
            err = ((self._ae_reconstruct(x) - x) / self.AE_RES_SD) ** 2
            s = np.full(T, np.nan); s[K - 1:] = err[:, :, self.CONT].mean(axis=(1, 2))
            return smooth(s)
        if name == "Isolation Forest":
            W = sliding_window_view(z, K, axis=0).transpose(0, 2, 1)
            s = np.full(T, np.nan); s[K - 1:] = -self.iforest.score_samples(self._if_features(W))
            return smooth(s)
        W = sliding_window_view(z, K, axis=0).transpose(0, 2, 1)[:T - K]
        P, S = self.predict(name, np.ascontiguousarray(W))
        rsd = self.RES_SD[name]
        acc, cnt = np.zeros(T), np.zeros(T)
        n = np.arange(len(P))
        for h in range(1, H + 1):
            tau = K + n - 1 + h
            ok = tau < T
            scale = S[n[ok], h - 1] if S is not None else rsd[h - 1]
            r = (z[tau[ok]] - (z[K + n[ok] - 1] + P[n[ok], h - 1])) / scale
            acc[tau[ok]] += (r[:, self.CONT] ** 2).mean(1)
            cnt[tau[ok]] += 1
        return smooth(np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan))

    def set_thresholds(self, budget=0.05):
        """Alarm threshold per method: only `budget` of clean validation recordings raise an alarm."""
        self.clean_test = {m: [self.score(m, self.Z[i]) for i in self.TEST] for m in METHODS}
        val_max = {m: [np.nanmax(self.score(m, self.Z[i])) for i in self.VAL] for m in METHODS}
        self.THRESH = {m: float(np.quantile(v, 1 - budget)) for m, v in val_max.items()}

    # ---------------------------------------------------------------- failures
    def inject(self, z, kind, rng, size=1.0):
        K, CONT, STEP_SD = self.K, self.CONT, self.STEP_SD
        z = z.copy()
        T, L = len(z), FAIL_LEN[kind]
        lo, hi = K + 5, T - L - 5
        if kind == "freeze":
            speed = np.linalg.norm(np.diff(z[:, CONT], axis=0), axis=1)
            cand = [t for t in range(lo, hi) if speed[t - 1] > np.median(speed)]
            t0 = int(rng.choice(cand)) if cand else int(rng.integers(lo, hi))
        else:
            t0 = int(rng.integers(lo, hi))
        dims = rng.choice(CONT, size=3, replace=False)
        sign = rng.choice([-1.0, 1.0], size=3)
        if kind == "jerk":
            z[t0:t0 + L, dims] += size * 5 * STEP_SD[dims] * sign
        elif kind == "freeze":
            z[t0:t0 + L] = z[t0 - 1]
        elif kind == "drift":
            z[t0:t0 + L, dims] += np.linspace(0, 1, L)[:, None] * size * 10 * STEP_SD[dims] * sign
        elif kind == "noise":
            z[t0:t0 + L, CONT] += rng.normal(0, 1, (L, len(CONT))) * size * 3 * STEP_SD[CONT]
        label = np.zeros(T, bool); label[t0:t0 + L] = True
        return z.astype(np.float32), label, t0

    def make_failures(self, size=1.0, kinds=FAILURES):
        rng = np.random.default_rng(1000 + self.seed)
        return {k: [(i, *self.inject(self.Z[i], k, rng, size)) for i in self.TEST] for k in kinds}

    def evaluate(self, failures, methods=METHODS):
        rows = []
        for m in methods:
            clean = self.clean_test[m]
            false_alarm = float(np.mean([np.nanmax(s) > self.THRESH[m] for s in clean]))
            alarm_steps = float(np.mean([np.nansum(s > self.THRESH[m]) for s in clean]))
            clean_sc = [s[~np.isnan(s)] for s in clean]
            for kind, eps in failures.items():
                sc, lb = list(clean_sc), [np.zeros(len(x), bool) for x in clean_sc]
                hits, delays = [], []
                for i, zd, lab, t0 in eps:
                    s = self.score(m, zd)
                    ok = ~np.isnan(s)
                    sc.append(s[ok]); lb.append(lab[ok])
                    alarm = np.where(s[t0:t0 + FAIL_LEN[kind] + 3] > self.THRESH[m])[0]
                    hits.append(len(alarm) > 0)
                    if len(alarm):
                        delays.append(alarm[0])
                sc, lb = np.concatenate(sc), np.concatenate(lb)
                rows.append({"seed": self.seed, "method": m, "failure": kind, "ROC-AUC": roc_auc(sc, lb),
                             "detection rate": float(np.mean(hits)),
                             "delay": float(np.mean(delays)) if delays else np.nan,
                             "false alarms @90% caught": false_alarm_rate_at_recall(sc, lb, 0.9),
                             "clean recordings with alarm": false_alarm,
                             "alarm steps per clean recording": alarm_steps})
        return pd.DataFrame(rows)

    def forecast_errors(self):
        rows = []
        for m in ("Rule: stays still", "Rule: keeps its speed", "GRU forecaster",
                  "Transformer forecaster", "Uncertainty-aware GRU"):
            P, _ = self.predict(m, self.Xte)
            mm = np.linalg.norm((P[:, :, self.XYZ] - self.Yte[:, :, self.XYZ]) * self.SD[self.XYZ], axis=-1) * 1000
            rows.append({"seed": self.seed, "method": m, "hand error 1 step [mm]": float(mm[:, 0].mean()),
                         f"hand error {self.H} steps [mm]": float(mm[:, -1].mean())})
        return pd.DataFrame(rows)

    def run(self, sizes=(0.5,)):
        """Everything for one seed: train, thresholds, failure tests. Returns result tables."""
        t0 = time.time()
        self.train_all()
        self.set_thresholds()
        det = self.evaluate(self.make_failures())
        small = []
        for size in sizes:
            r = self.evaluate(self.make_failures(size, kinds=["jerk", "drift", "noise"]))
            r["size"] = size
            small.append(r)
        self.log(f"  seed {self.seed} done in {time.time() - t0:.0f}s")
        return {"forecast": self.forecast_errors(), "detection": det, "small": pd.concat(small)}
