"""
Short-circuit (shoot-through) event and the gate driver's DESAT protection.

The low-side device turns on straight across the DC link through only the
commutation-loop inductance - the worst, fastest-rising fault. The gate
driver's desaturation (DESAT) circuit notices that Vds stays high while the
gate is on, and after a fixed chain of delays starts a *soft* turn-off with a
small constant current, so the huge fault current falls slowly enough not to
create a destructive L*di/dt overshoot.

Timing chain (UCC21750-Q1 datasheet values, typ / worst):
    leading-edge blanking            200 ns
    blanking capacitor charge        C_blk * V_th / I_chg
                                     V_th 9.15 V (8.5-9.8), I_chg 500 uA (430-570)
    DESAT deglitch filter            140 ns (50-230)
    DESAT -> output low              200 ns (150-300)
    soft turn-off current            400 mA (250-570)
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .device import SiCMosfet
from . import spice

_trapz = getattr(np, "trapezoid", None) or np.trapz

DESAT_TYP = {"t_leb": 200e-9, "v_th": 9.15, "i_chg": 500e-6, "t_deglitch": 140e-9,
             "t_prop": 200e-9, "i_sto": 0.400}
DESAT_WORST = {"t_leb": 200e-9, "v_th": 9.8, "i_chg": 430e-6, "t_deglitch": 230e-9,
               "t_prop": 300e-9, "i_sto": 0.250}


def desat_delay(c_blk: float, p: dict = DESAT_TYP) -> float:
    """Time from gate turn-on until the driver starts the soft turn-off (s)."""
    return p["t_leb"] + c_blk * p["v_th"] / p["i_chg"] + p["t_deglitch"] + p["t_prop"]


@dataclass
class SCConfig:
    vdc: float = 850.0
    tj: float = 25.0
    l_sc: float = 32e-9           # loop inductance seen by one device (per-device equivalent)
    r_damp: float = 100.0
    rg_on: float = 12.0           # total turn-on gate resistance per device (ext + driver share)
    vgs_on: float = 15.0
    vgs_off: float = -4.0
    edge: float = 5e-9
    t_on: float = 0.2e-6          # gate turn-on command
    t_trip: float = 0.8e-6        # soft turn-off start, measured from t_on
    i_sto: float = 0.1            # soft turn-off current per device, A
    t_after: float = 3.0e-6
    tmax: float = 0.5e-9

    def with_(self, **kw):
        return replace(self, **kw)


def netlist(dev: SiCMosfet, cfg: SCConfig) -> str:
    t_off = cfg.t_on + cfg.t_trip
    t_stop = t_off + cfg.t_after
    on, off, e = cfg.vgs_on, cfg.vgs_off, cfg.edge
    return f"""* Short circuit with DESAT soft turn-off
.options temp=27 tnom=27 method=gear reltol=1e-3 abstol=1e-5 vntol=1e-4 chgtol=1e-13 itl4=200
{dev.subckt(cfg.tj, name="DEV")}
Vdc bus 0 {cfg.vdc:.6g}
Lsc bus d {cfg.l_sc:.6g}
Rsc bus d {cfg.r_damp:.6g}
Vid d dd 0
X1 dd g ks 0 DEV
* driver: on-pulse through Rg; at the trip the output stage disconnects
Vsrc src ks PWL(0 {off} {cfg.t_on} {off} {cfg.t_on + e} {on})
Sdrv src gdrv ctl 0 SDRV
.model SDRV SW(VT=0.5 VH=0.1 RON=1e-3 ROFF=1e9)
Vctl ctl 0 PWL(0 1 {t_off} 1 {t_off + 1e-9} 0)
Rg gdrv g {cfg.rg_on:.6g}
* soft turn-off: constant current sink to VEE, clamped at VEE
Vee vee ks {off}
Isto g vee PWL(0 0 {t_off} 0 {t_off + 1e-9} {cfg.i_sto:.6g})
Dclamp vee g DCLAMP
.model DCLAMP D(IS=1e-14 N=1)
.tran 0.1n {t_stop:.6g} 0 {cfg.tmax:.6g}
.end
"""


def run(dev: SiCMosfet, cfg: SCConfig) -> tuple[dict, dict]:
    r = spice.run(netlist(dev, cfg), ["v(dd,ks)", "i(vid)", "v(g,ks)"])
    t, vds, i_d, vgs = r["time"], r["v(dd,ks)"], r["i(vid)"], r["v(g,ks)"]
    t_off = cfg.t_on + cfg.t_trip
    m = t >= cfg.t_on
    i_trip = float(np.interp(t_off, t, i_d))
    after = t > t_off
    end_idx = np.argmax(after & (i_d < 1.0))
    t_end = t[end_idx] if after[end_idx] and i_d[end_idx] < 1.0 else np.nan
    sel = m & (t <= (t_end if np.isfinite(t_end) else t[-1]))
    return ({
        "i_peak_A": float(i_d[m].max()),
        "i_at_trip_A": i_trip,
        "t_trip_us": cfg.t_trip * 1e6,
        "t_fault_total_us": (t_end - cfg.t_on) * 1e6 if np.isfinite(t_end) else np.nan,
        "e_sc_mJ": float(_trapz((vds * i_d)[sel], t[sel])) * 1e3,
        "v_peak_off_V": float(vds[after].max()),
    }, {"t": t, "vds": vds, "id": i_d, "vgs": vgs})
