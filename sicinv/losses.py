"""
Inverter loss model: one switch position of a 3-phase, 2-level inverter,
averaged over one electrical (fundamental) period.

Phase current  i(theta) = sqrt(2) * I_rms * sin(theta)        (+ = out of the leg)
Upper duty     d(theta) = 0.5 * (1 + m * sin(theta + phi))    (m = modulation index,
                                                                phi = power-factor angle)
Each switch position = n_par identical devices sharing the current equally.

SiC MOSFET position (synchronous rectification: the channel conducts in both
directions whenever the gate is on, the body diode only during dead time):
    conduction   d * Vds_on(i/n) * |i|
    dead time    [i < 0] * 2 t_dead f_sw * Vsd(i/n) * |i|
    switching    [i > 0] * f_sw * n * (Eon + Eoff)(i/n)
    recovery     [i < 0] * f_sw * n * Err(i/n)

IGBT + diode position (an IGBT cannot conduct backwards, so the diode takes
negative current):
    IGBT   [i > 0] * (d * Vce(i/n) * |i| + f_sw * n * (Eon + Eoff)(i/n))
    diode  [i < 0] * (d * Vf(i/n) * |i| + 2 t_dead f_sw Vf |i| + f_sw * n * Erec(i/n))

The lower switch sees the mirror image, so it has the same average losses;
the whole inverter is 6 x one position.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .device import SiCMosfet

THETA = np.linspace(0, 2 * np.pi, 720, endpoint=False)
T_RUNAWAY = 300.0   # C; above this the electro-thermal iteration is declared a runaway


def _waveforms(i_rms, m, phi):
    i = np.sqrt(2) * i_rms * np.sin(THETA)
    d = 0.5 * (1 + m * np.sin(THETA + phi))
    return i, np.abs(i), d


# --------------------------------------------------------------------- SiC --
@dataclass
class EnergyTable:
    """Per-device switching energies from the double-pulse simulations (J)."""
    i: np.ndarray            # A, per device
    tj: np.ndarray           # C
    eon: np.ndarray          # shape (len(tj), len(i)) at v_ref
    eoff: np.ndarray
    err: np.ndarray
    v_ref: float
    kv_on: float = 1.0       # E ~ (Vdc / v_ref) ** kv
    kv_off: float = 1.0
    kv_rr: float = 1.0
    meta: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "EnergyTable":
        d = json.loads(Path(path).read_text())
        arr = {k: np.array(d[k]) for k in ("i", "tj", "eon", "eoff", "err")}
        return cls(**arr, v_ref=d["v_ref"], kv_on=d["kv_on"], kv_off=d["kv_off"],
                   kv_rr=d["kv_rr"], meta=d.get("meta", {}))

    def _interp(self, grid, i, tj):
        f = RegularGridInterpolator((self.tj, self.i), grid, bounds_error=False, fill_value=None)
        pts = np.column_stack([np.full_like(i, np.clip(tj, self.tj[0], self.tj[-1])),
                               np.clip(i, 0, None)])
        return np.clip(f(pts), 0, None)

    def energies(self, i, tj, vdc):
        i = np.asarray(i, float)
        s = vdc / self.v_ref
        return (self._interp(self.eon, i, tj) * s ** self.kv_on,
                self._interp(self.eoff, i, tj) * s ** self.kv_off,
                self._interp(self.err, i, tj) * s ** self.kv_rr)


@dataclass
class SiCPosition:
    dev: SiCMosfet
    etab: EnergyTable
    n_par: int = 4
    t_dead: float = 200e-9

    def __post_init__(self):
        self._vds = lru_cache(maxsize=512)(self._vds_table)

    def _vds_table(self, tj_key: int):
        grid = np.linspace(0, 400, 81)
        return grid, np.array([0.0] + [self.dev.vds_on(x, tj_key) for x in grid[1:]])

    def vds_on(self, i_dev, tj):
        grid, v = self._vds(int(round(tj)))
        return np.interp(i_dev, grid, v)

    def losses(self, i_rms, vdc, fsw, tj, m=0.9, phi=0.0) -> dict:
        """Average losses (W) of one switch position = n_par devices."""
        i, ia, d = _waveforms(i_rms, m, phi)
        idev = ia / self.n_par
        pos, neg = i > 0, i < 0
        eon, eoff, err = self.etab.energies(idev, tj, vdc)
        p = {
            "conduction": np.mean(d * self.vds_on(idev, tj) * ia),
            "dead_time": np.mean(neg * 2 * self.t_dead * fsw * self.dev.vsd(np.maximum(idev, 1e-3), tj) * ia),
            "turn_on": np.mean(pos * fsw * self.n_par * eon),
            "turn_off": np.mean(pos * fsw * self.n_par * eoff),
            "recovery": np.mean(neg * fsw * self.n_par * err),
        }
        p = {k: float(v) for k, v in p.items()}
        p["total"] = sum(p.values())
        return p

    def steady_state(self, i_rms, vdc, fsw, rth_dev, t_cool, m=0.9, phi=0.0, iters=30):
        """Electro-thermal fixed point: losses depend on Tj, Tj depends on losses."""
        tj = t_cool + 20.0
        for _ in range(iters):
            p = self.losses(i_rms, vdc, fsw, tj, m, phi)
            tj_new = t_cool + p["total"] / self.n_par * rth_dev
            if tj_new > T_RUNAWAY:          # losses grow faster than cooling: no solution
                p["tj"], p["runaway"] = float("inf"), True
                return p
            if abs(tj_new - tj) < 0.05:
                break
            tj = 0.5 * (tj + tj_new)
        p["tj"], p["runaway"] = tj_new, False
        return p


# -------------------------------------------------------------------- IGBT --
@dataclass
class IGBTPosition:
    ds: dict                 # parsed iky75n120ch7_datasheet.json
    n_par: int = 4
    t_dead: float = 1.0e-6   # IGBTs need a longer dead time (tail current, slower turn-off)
    kv: float = 1.0          # linear voltage scaling of energies (favours the IGBT)

    def _curve(self, key, i, tj):
        c25 = np.array(self.ds[key]["25C"]).T
        c175 = np.array(self.ds[key]["175C"]).T
        w = np.clip((tj - 25) / 150, 0, 1)
        return (1 - w) * np.interp(i, *c25) + w * np.interp(i, *c175)

    def _energy(self, name, i, tj, vdc):
        e = np.array(self.ds["energy_vs_ic_175C_600V"][name]).T
        e175 = np.interp(i, e[0], e[1], right=None)
        sc = self.ds["scalars"]
        ratio25 = sc[f"{name}_mJ_25C"] / sc[f"{name}_mJ_175C"]
        w = np.clip((tj - 25) / 150, 0, 1)
        return 1e-3 * e175 * ((1 - w) * ratio25 + w) * (vdc / 600.0) ** self.kv

    def losses(self, i_rms, vdc, fsw, tj_igbt, tj_diode, m=0.9, phi=0.0) -> dict:
        i, ia, d = _waveforms(i_rms, m, phi)
        idev = ia / self.n_par
        pos, neg = i > 0, i < 0
        n = self.n_par
        p = {
            "igbt_conduction": np.mean(pos * d * self._curve("vcesat_curves", idev, tj_igbt) * ia),
            "igbt_turn_on": np.mean(pos * fsw * n * self._energy("Eon", idev, tj_igbt, vdc)),
            "igbt_turn_off": np.mean(pos * fsw * n * self._energy("Eoff", idev, tj_igbt, vdc)),
            "diode_conduction": np.mean(neg * (d + 2 * self.t_dead * fsw)
                                        * self._curve("diode_vf_curves", idev, tj_diode) * ia),
            "diode_recovery": np.mean(neg * fsw * n * self._energy("Erec", idev, tj_diode, vdc)),
        }
        p = {k: float(v) for k, v in p.items()}
        p["total"] = sum(p.values())
        p["igbt"] = p["igbt_conduction"] + p["igbt_turn_on"] + p["igbt_turn_off"]
        p["diode"] = p["diode_conduction"] + p["diode_recovery"]
        return p

    def steady_state(self, i_rms, vdc, fsw, r_cs, t_cool, m=0.9, phi=0.0, iters=40):
        sc = self.ds["scalars"]
        tj_i = tj_d = t_cool + 20.0
        for _ in range(iters):
            p = self.losses(i_rms, vdc, fsw, tj_i, tj_d, m, phi)
            pi, pd = p["igbt"] / self.n_par, p["diode"] / self.n_par
            t_case = t_cool + (pi + pd) * r_cs
            ti_new = t_case + pi * sc["Rth_jc_igbt_K_per_W"]
            td_new = t_case + pd * sc["Rth_jc_diode_K_per_W"]
            if max(ti_new, td_new) > T_RUNAWAY:
                p["tj"] = p["tj_igbt"] = p["tj_diode"] = float("inf")
                p["runaway"] = True
                return p
            if abs(ti_new - tj_i) < 0.05 and abs(td_new - tj_d) < 0.05:
                break
            tj_i, tj_d = 0.5 * (tj_i + ti_new), 0.5 * (tj_d + td_new)
        p["runaway"] = False
        p["tj"] = max(ti_new, td_new)
        p["tj_igbt"], p["tj_diode"] = ti_new, td_new
        return p
