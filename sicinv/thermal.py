"""
Thermal models.

A Foster network is a sum of first-order RC terms, Zth(t) = sum r_i (1 - e^(-t/tau_i)).
It is what datasheets give for junction-to-case, and it is exact to step forward
in time when the power is constant over each step:

    dT_i(t + dt) = dT_i(t) * e^(-dt/tau_i) + P * r_i * (1 - e^(-dt/tau_i))

Stack: junction -> case (device datasheet) -> coolant (thermal pad + cold plate).
Putting Foster networks in series is an approximation (a physically exact
stack needs Cauer form); it is the usual first-pass simplification and it is
exact in steady state.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares


@dataclass
class Foster:
    r: list[float]           # K/W
    tau: list[float]         # s

    @property
    def rth(self) -> float:
        return float(sum(self.r))

    def zth(self, t):
        t = np.asarray(t, float)[..., None]
        return np.sum(np.array(self.r) * (1 - np.exp(-t / np.array(self.tau))), axis=-1)


def fit_foster(t, z, n: int = 4) -> Foster:
    """Fit an n-term Foster network to a single-pulse Zth curve (log error)."""
    t, z = np.asarray(t, float), np.asarray(z, float)
    tau0 = np.geomspace(t.min() * 3, t.max() / 3, n)
    r0 = np.full(n, z.max() / n)

    def res(p):
        f = Foster(list(np.exp(p[:n])), list(np.exp(p[n:])))
        return np.log(f.zth(t) / z)

    fit = least_squares(res, np.log(np.concatenate([r0, tau0])))
    return Foster(list(np.exp(fit.x[:n])), list(np.exp(fit.x[n:])))


@dataclass
class ThermalStack:
    """Junction -> case -> coolant for ONE device; coolant temperature is fixed."""
    jc: Foster
    cs: Foster                            # case -> coolant (pad + cold plate)
    t_coolant: float = 65.0
    _state: np.ndarray = field(default=None, repr=False)

    @property
    def rth_total(self) -> float:
        return self.jc.rth + self.cs.rth

    def steady_tj(self, p: float) -> float:
        return self.t_coolant + p * self.rth_total

    def reset(self):
        self._state = np.zeros(len(self.jc.r) + len(self.cs.r))

    def step(self, p: float, dt: float) -> float:
        """Advance by dt with constant power p (W); returns Tj (C)."""
        if self._state is None:
            self.reset()
        r = np.array(self.jc.r + self.cs.r)
        tau = np.array(self.jc.tau + self.cs.tau)
        a = np.exp(-dt / tau)
        self._state = self._state * a + p * r * (1 - a)
        return self.t_coolant + float(self._state.sum())


def coldplate_path(r_pad: float = 0.22, r_plate: float = 0.30,
                   tau_pad: float = 0.08, tau_plate: float = 4.0) -> Foster:
    """Case -> coolant for a TO-247 on an insulating pad and a liquid cold plate.

    r_pad  : 0.2 mm insulating pad, k ~ 3 W/mK, ~3 cm^2   -> ~0.22 K/W
    r_plate: spreading + convection share of the cold plate -> ~0.30 K/W
    """
    return Foster([r_pad, r_plate], [tau_pad, tau_plate])
