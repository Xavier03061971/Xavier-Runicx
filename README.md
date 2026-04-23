from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np


# =========================================================
# BENCHMARK FUNCTIONS
# =========================================================

def sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1.0 - x[:-1])**2))


def rastrigin(x: np.ndarray) -> float:
    return float(10.0 * len(x) + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))


def ackley(x: np.ndarray) -> float:
    d = len(x)
    s1 = np.sum(x**2)
    s2 = np.sum(np.cos(2.0 * np.pi * x))
    return float(
        -20.0 * np.exp(-0.2 * np.sqrt(s1 / d))
        - np.exp(s2 / d)
        + 20.0
        + math.e
    )


FUNCTIONS: Dict[str, Callable[[np.ndarray], float]] = {
    "sphere": sphere,
    "rosenbrock": rosenbrock,
    "rastrigin": rastrigin,
    "ackley": ackley,
}

BOUNDS: Dict[str, Tuple[float, float]] = {
    "sphere": (-5.0, 5.0),
    "rosenbrock": (-3.0, 3.0),
    "rastrigin": (-5.12, 5.12),
    "ackley": (-5.0, 5.0),
}


# =========================================================
# HELPERS
# =========================================================

def clip(x: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(x, low, high)


def diversity(X: np.ndarray) -> float:
    return float(np.mean(np.std(X, axis=0)))


def safe_choice_excluding(
    rng: np.random.Generator,
    pool_size: int,
    exclude: set[int],
    k: int,
) -> List[int]:
    candidates = [i for i in range(pool_size) if i not in exclude]
    picks = rng.choice(candidates, size=k, replace=False)
    return [int(x) for x in picks]


def lehmer_mean(values: List[float], weights: List[float], default: float) -> float:
    if not values:
        return default
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    den = np.sum(w * v)
    if abs(den) < 1e-12:
        return default
    num = np.sum(w * v * v)
    return float(num / den)


# =========================================================
# RUNIC X
# =========================================================

@dataclass
class RuneOp:
    op: str
    value: object


class RunicXParser:
    @staticmethod
    def parse(text: str) -> List[RuneOp]:
        ops: List[RuneOp] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            p = line.split()
            head = p[0].upper()
            tail = p[1:]

            if head == "TARGET":
                ops.append(RuneOp("TARGET", tail[0].lower()))
            elif head == "DIM":
                ops.append(RuneOp("DIM", int(tail[0])))
            elif head == "POP":
                ops.append(RuneOp("POP", int(tail[0])))
            elif head == "ITERS":
                ops.append(RuneOp("ITERS", int(tail[0])))
            elif head == "P":
                ops.append(RuneOp("P", float(tail[0])))
            elif head == "H":
                ops.append(RuneOp("H", int(tail[0])))
            elif head == "MU_F":
                ops.append(RuneOp("MU_F", float(tail[0])))
            elif head == "MU_CR":
                ops.append(RuneOp("MU_CR", float(tail[0])))
            elif head == "RESTART_AFTER":
                ops.append(RuneOp("RESTART_AFTER", int(tail[0])))
            elif head == "NOISE":
                ops.append(RuneOp("NOISE", float(tail[0])))
            elif head == "ARCHIVE":
                ops.append(RuneOp("ARCHIVE", int(tail[0])))
            else:
                raise ValueError(f"Instrucción desconocida: {line}")
        return ops


# =========================================================
# CONFIG
# =========================================================

@dataclass
class XavierConfig:
    target: str = "rastrigin"
    dim: int = 10
    pop: int = 60
    iters: int = 600
    p: float = 0.20          # fracción p-best
    H: int = 8               # tamaño de memorias SHADE
    mu_f: float = 0.60
    mu_cr: float = 0.85
    restart_after: int = 60
    noise: float = 0.0
    archive: int = 200

    @classmethod
    def from_runic(cls, text: str) -> "XavierConfig":
        cfg = cls()
        for op in RunicXParser.parse(text):
            if op.op == "TARGET":
                cfg.target = str(op.value)
            elif op.op == "DIM":
                cfg.dim = int(op.value)
            elif op.op == "POP":
                cfg.pop = int(op.value)
            elif op.op == "ITERS":
                cfg.iters = int(op.value)
            elif op.op == "P":
                cfg.p = float(op.value)
            elif op.op == "H":
                cfg.H = int(op.value)
            elif op.op == "MU_F":
                cfg.mu_f = float(op.value)
            elif op.op == "MU_CR":
                cfg.mu_cr = float(op.value)
            elif op.op == "RESTART_AFTER":
                cfg.restart_after = int(op.value)
            elif op.op == "NOISE":
                cfg.noise = float(op.value)
            elif op.op == "ARCHIVE":
                cfg.archive = int(op.value)
        return cfg


# =========================================================
# OUTPUT
# =========================================================

@dataclass
class RunResult:
    best_x: np.ndarray
    best_value: float
    history: List[float]
    elapsed: float


# =========================================================
# XAVIER CORE (JADE / SHADE-LITE)
# =========================================================

class Xavier:
    def __init__(self, cfg: XavierConfig, seed: int = 42):
        self.cfg = cfg
        self.func = FUNCTIONS[cfg.target]
        self.low, self.high = BOUNDS[cfg.target]
        self.dim = cfg.dim
        self.pop = cfg.pop
        self.iters = cfg.iters
        self.span = self.high - self.low

        self.rng = np.random.default_rng(seed)

        self.X = self.rng.uniform(self.low, self.high, size=(self.pop, self.dim))
        self.values = np.array([self.func(x) for x in self.X], dtype=float)

        best_idx = int(np.argmin(self.values))
        self.best_x = self.X[best_idx].copy()
        self.best_value = float(self.values[best_idx])

        self.history: List[float] = [self.best_value]
        self.archive: List[np.ndarray] = []
        self.no_improve_steps = 0

        self.M_F = np.full(self.cfg.H, self.cfg.mu_f, dtype=float)
        self.M_CR = np.full(self.cfg.H, self.cfg.mu_cr, dtype=float)
        self.mem_index = 0

    # -----------------------------------------------------
    # state
    # -----------------------------------------------------

    def pattern(self) -> str:
        if len(self.history) < 12:
            return "unknown"

        h = np.array(self.history[-12:], dtype=float)
        d = np.diff(h)
        mean_d = float(np.mean(d))
        std_d = float(np.std(d))

        if mean_d < 0 and std_d < 1e-8:
            return "stable"
        if abs(mean_d) < 1e-6:
            return "stalled"
        if mean_d > 0:
            return "divergent"
        return "mixed"

    def current_p(self) -> float:
        pat = self.pattern()
        if self.cfg.target == "rastrigin":
            if pat == "stalled":
                return 0.35
            if pat == "stable":
                return 0.10
            return self.cfg.p
        if self.cfg.target == "ackley":
            if pat == "stalled":
                return 0.28
            if pat == "stable":
                return 0.12
            return self.cfg.p
        if self.cfg.target == "sphere":
            if pat == "stable":
                return 0.05
            return 0.12
        if self.cfg.target == "rosenbrock":
            if pat == "stable":
                return 0.06
            return 0.14
        return self.cfg.p

    def sample_F_CR(self, idx_mem: int) -> Tuple[float, float]:
        mu_f = self.M_F[idx_mem]
        mu_cr = self.M_CR[idx_mem]

        F = -1.0
        while F <= 0.0:
            F = mu_f + 0.10 * np.tan(math.pi * (self.rng.random() - 0.5))
        F = min(F, 1.0)

        CR = float(self.rng.normal(mu_cr, 0.10))
        CR = min(1.0, max(0.0, CR))

        pat = self.pattern()
        if pat == "stalled":
            F = min(1.0, F * 1.10)
            CR = min(1.0, CR + 0.05)
        elif pat == "stable":
            F = max(0.20, F * 0.85)

        return F, CR

    def add_archive(self, x: np.ndarray) -> None:
        self.archive.append(x.copy())
        if len(self.archive) > self.cfg.archive:
            idx = int(self.rng.integers(0, len(self.archive)))
            del self.archive[idx]

    def restart_partial(self) -> None:
        elite_n = max(3, self.pop // 5)
        elite_idx = np.argsort(self.values)[:elite_n]
        elite = self.X[elite_idx]

        if self.cfg.target in {"sphere", "rosenbrock"}:
            local = 0.10
            sigma = 0.02
        elif self.cfg.target == "ackley":
            local = 0.18
            sigma = 0.04
        else:
            local = 0.28
            sigma = 0.08

        keep = set(elite_idx.tolist())

        for i in range(self.pop):
            if i in keep:
                continue
            anchor = elite[int(self.rng.integers(0, len(elite)))]
            x = anchor + self.rng.uniform(-local, local, size=self.dim) * self.span
            x += self.rng.normal(0.0, sigma * self.span, size=self.dim)
            self.X[i] = clip(x, self.low, self.high)

        self.values = np.array([self.func(x) for x in self.X], dtype=float)
        best_idx = int(np.argmin(self.values))
        if self.values[best_idx] < self.best_value:
            self.best_value = float(self.values[best_idx])
            self.best_x = self.X[best_idx].copy()

    # -----------------------------------------------------
    # trial generation
    # -----------------------------------------------------

    def generate_trial(self, i: int, F: float, CR: float, pbest_idx: int) -> np.ndarray:
        exclude = {i, pbest_idx}
        r1, = safe_choice_excluding(self.rng, self.pop, exclude, 1)

        use_archive = len(self.archive) > 0 and self.rng.random() < 0.5
        if use_archive:
            r2_x = self.archive[int(self.rng.integers(0, len(self.archive)))]
        else:
            exclude2 = {i, pbest_idx, r1}
            r2, = safe_choice_excluding(self.rng, self.pop, exclude2, 1)
            r2_x = self.X[r2]

        xi = self.X[i]
        xp = self.X[pbest_idx]
        xr1 = self.X[r1]

        donor = xi + F * (xp - xi) + F * (xr1 - r2_x)

        if self.cfg.noise > 0.0:
            donor += self.rng.normal(0.0, self.cfg.noise * self.span, size=self.dim)

        donor = clip(donor, self.low, self.high)

        jrand = int(self.rng.integers(0, self.dim))
        trial = xi.copy()
        mask = self.rng.random(self.dim) < CR
        mask[jrand] = True
        trial[mask] = donor[mask]

        return clip(trial, self.low, self.high)

    # -----------------------------------------------------
    # adaptation
    # -----------------------------------------------------

    def update_memories(self, SF: List[float], SCR: List[float], dF: List[float]) -> None:
        if not SF:
            return

        new_mf = lehmer_mean(SF, dF, self.M_F[self.mem_index])
        if SCR:
            weights = np.asarray(dF[:len(SCR)], dtype=float)
            cr_arr = np.asarray(SCR, dtype=float)
            den = np.sum(weights)
            new_mcr = self.M_CR[self.mem_index] if den < 1e-12 else float(np.sum(weights * cr_arr) / den)
        else:
            new_mcr = self.M_CR[self.mem_index]

        self.M_F[self.mem_index] = min(0.99, max(0.05, new_mf))
        self.M_CR[self.mem_index] = min(0.99, max(0.00, new_mcr))
        self.mem_index = (self.mem_index + 1) % self.cfg.H

    # -----------------------------------------------------
    # run
    # -----------------------------------------------------

    def run(self) -> RunResult:
        t0 = time.time()

        for t in range(self.iters):
            p = self.current_p()
            p_count = max(2, int(math.ceil(p * self.pop)))
            sorted_idx = np.argsort(self.values)
            top_idx = sorted_idx[:p_count]

            new_X = self.X.copy()
            new_values = self.values.copy()

            SF: List[float] = []
            SCR: List[float] = []
            dF: List[float] = []

            for i in range(self.pop):
                mem_k = int(self.rng.integers(0, self.cfg.H))
                F, CR = self.sample_F_CR(mem_k)
                pbest_idx = int(top_idx[int(self.rng.integers(0, len(top_idx)))])

                trial = self.generate_trial(i, F, CR, pbest_idx)
                trial_val = float(self.func(trial))

                if trial_val <= self.values[i]:
                    improvement = abs(self.values[i] - trial_val)
                    self.add_archive(self.X[i])

                    new_X[i] = trial
                    new_values[i] = trial_val

                    SF.append(F)
                    SCR.append(CR)
                    dF.append(improvement)

            self.X = new_X
            self.values = new_values

            best_idx = int(np.argmin(self.values))
            current_best = float(self.values[best_idx])

            if current_best < self.best_value:
                self.best_value = current_best
                self.best_x = self.X[best_idx].copy()
                self.no_improve_steps = 0
            else:
                self.no_improve_steps += 1

            self.update_memories(SF, SCR, dF)

            if self.no_improve_steps >= self.cfg.restart_after:
                self.restart_partial()
                self.no_improve_steps = 0

            self.history.append(self.best_value)
            self.history = self.history[-500:]

            if t % 10 == 0 or t == self.iters - 1:
                print(
                    f"[{t:03d}] "
                    f"best={self.best_value:.10f} "
                    f"pattern={self.pattern():<9} "
                    f"div={diversity(self.X):.5f}"
                )

        return RunResult(
            best_x=self.best_x.copy(),
            best_value=self.best_value,
            history=list(self.history),
            elapsed=time.time() - t0,
        )


# =========================================================
# RUNIC PROGRAMS
# =========================================================

RUNIC_SPHERE = """
TARGET sphere
DIM 10
POP 50
ITERS 500
P 0.10
H 8
MU_F 0.55
MU_CR 0.92
RESTART_AFTER 80
NOISE 0.0000
ARCHIVE 200
"""

RUNIC_ROSENBROCK = """
TARGET rosenbrock
DIM 10
POP 70
ITERS 900
P 0.12
H 10
MU_F 0.58
MU_CR 0.90
RESTART_AFTER 120
NOISE 0.0000
ARCHIVE 300
"""

RUNIC_RASTRIGIN = """
TARGET rastrigin
DIM 10
POP 90
ITERS 1200
P 0.22
H 10
MU_F 0.72
MU_CR 0.94
RESTART_AFTER 60
NOISE 0.0000
ARCHIVE 500
"""

RUNIC_ACKLEY = """
TARGET ackley
DIM 10
POP 70
ITERS 900
P 0.18
H 10
MU_F 0.66
MU_CR 0.92
RESTART_AFTER 70
NOISE 0.0000
ARCHIVE 300
"""


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    programs = {
        "Sphere": RUNIC_SPHERE,
        "Rosenbrock": RUNIC_ROSENBROCK,
        "Rastrigin": RUNIC_RASTRIGIN,
        "Ackley": RUNIC_ACKLEY,
    }

    for name, program in programs.items():
        print(f"\n=== {name} ===")
        cfg = XavierConfig.from_runic(program)
        engine = Xavier(cfg, seed=42)
        result = engine.run()
        print("final best:", result.best_value)
        print("best x:", result.best_x)
        print("time:", result.elapsed)
