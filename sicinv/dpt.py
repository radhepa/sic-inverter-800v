"""
Double-pulse test (DPT) bench, the standard way power-device switching
losses are measured, built as an ngspice netlist.

    +800 V ──┬──────────────┬─────────────┐
             │              │             │
          L_loop/2      ┌───┴───┐      L_load (157 uH)
             │          │  HS   │ gate held OFF (-4 V); its body diode
             │          │ FET   │ carries the load current while DUT is off
             │          └───┬───┘         │
             │              ├── sw ───────┘
             │          ┌───┴───┐
           V_dc         │  LS   │ DUT, driven with two pulses:
             │          │ FET   │  pulse 1 ramps the inductor to I_test
             │          └───┬───┘  -> turn-OFF at I_test   (Eoff)
             │          L_loop/2   pulse 2 turns ON into I_test (Eon)
             └──────────────┘

Energies are integrated from the terminal waveforms exactly as on a real
bench: Eon from 10 % Vgs to Vds < 2 % Vdc, Eoff from 90 % Vgs to Id < 2 % I.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

from .device import SiCMosfet
from . import spice

_trapz = getattr(np, "trapezoid", None) or np.trapz


@dataclass
class DPTConfig:
    vdc: float = 800.0          # DC link voltage, V
    i_test: float = 50.0        # current at the switching instants, A
    tj: float = 25.0            # junction temperature, C
    rg_on: float = 2.5          # external gate resistor for turn-on, ohm
    rg_off: float = 2.5         # external gate resistor for turn-off, ohm
    r_drv: float = 0.7          # gate-driver output resistance, ohm
    r_drv_on: float | None = None   # if set: separate pull-up (source) resistance
    r_drv_off: float | None = None  # if set: separate pull-down (sink) resistance
    vgs_on: float = 15.0
    vgs_off: float = -4.0
    l_loop: float = 20e-9       # external power-loop inductance, H (package adds 2x(Ld+Ls))
    r_damp: float = 25.0        # resistor across each half of l_loop: models the
                                # frequency-dependent loop resistance (skin effect,
                                # capacitor ESR) that damps the ringing; no DC loss
    # high-frequency decoupling ("snubber") capacitor right at the switches:
    # the bulk DC link sits behind l_loop, the fast loop is only l_hf
    c_hf: float = 0.0           # F (0 = no local capacitor)
    l_hf: float = 0.0           # H, loop between that capacitor and the switches
    esl_hf: float = 1e-9        # H
    esr_hf: float = 0.02        # ohm
    l_load: float = 157e-6
    c_load: float = 10e-12      # load-inductor winding capacitance
    t0: float = 0.5e-6          # first rising edge
    t_gap: float = 2.0e-6       # off time between the pulses
    t_second: float = 1.0e-6    # length of pulse 2
    edge: float = 5e-9          # driver output rise/fall time
    tmax: float = 0.25e-9       # max simulator time step
    # the non-switching (high-side) device
    hs_vgs_off: float | None = None   # default: vgs_off
    hs_rg: float | None = None        # default: rg_off
    # paralleling: n_par devices per switch, each with its own gate resistor;
    # vth_offsets shifts individual low-side thresholds (part-to-part spread)
    n_par: int = 1
    vth_offsets: tuple = ()
    r_ks: float = 0.5                 # Kelvin-source resistor per device (n_par > 1)
    miller_clamp: bool = False        # active Miller clamp on the HS gate
    clamp_vth: float = 2.0            # clamp engages below this Vgs
    clamp_r: float = 0.5              # clamp on-resistance

    def with_(self, **kw) -> "DPTConfig":
        return replace(self, **kw)


def _t1(cfg: DPTConfig) -> float:
    """Length of the first pulse so the inductor current reaches I_test."""
    return cfg.i_test * cfg.l_load / cfg.vdc


# Solver settings, tried in order. Behavioural capacitors make the switching
# transients stiff; if a run stalls ("timestep too small") the next, slightly
# looser set is used. Tolerances stay far below the quantities measured
# (800 V, 50 A, mJ energies).
SOLVER_OPTIONS = [
    "method=gear reltol=1e-3 abstol=1e-5 vntol=1e-4 chgtol=1e-13",
    "method=gear reltol=2e-3 abstol=1e-4 vntol=1e-3 chgtol=1e-12",
    "method=trap reltol=1e-3 abstol=1e-5 vntol=1e-4 chgtol=1e-13",
]


def netlist(dev: SiCMosfet, cfg: DPTConfig, solver: str = SOLVER_OPTIONS[0]) -> str:
    t1 = _t1(cfg)
    e, t0 = cfg.edge, cfg.t0
    off, on = cfg.vgs_off, cfg.vgs_on
    a = t0 + t1                       # first turn-off command
    b = a + cfg.t_gap                 # second turn-on command
    c = b + cfg.t_second              # second turn-off command
    pwl = (f"0 {off} {t0} {off} {t0+e} {on} {a} {on} {a+e} {off} "
           f"{b} {off} {b+e} {on} {c} {on} {c+e} {off}")
    t_stop = c + 0.6e-6
    hs_off = cfg.vgs_off if cfg.hs_vgs_off is None else cfg.hs_vgs_off
    hs_rg = cfg.rg_off if cfg.hs_rg is None else cfg.hs_rg
    n = cfg.n_par
    offs = list(cfg.vth_offsets) + [0.0] * (n - len(cfg.vth_offsets))
    subs = [dev.subckt(cfg.tj, name="DEV")]
    lines = []
    for k in range(1, n + 1):
        name = "DEV"
        if offs[k - 1]:
            name = f"DEV{k}"
            subs.append(replace(dev, vt25=dev.vt25 + offs[k - 1],
                                vt175=dev.vt175 + offs[k - 1]).subckt(cfg.tj, name=name))
        hs_ks = "hs_kc" if n == 1 else f"hs_ks{k}"
        ls_ks = "ls_kc" if n == 1 else f"ls_ks{k}"
        lines += [
            f"XHS{k} hs_d hs_g{k} {hs_ks} sw DEV",
            f"Rghs{k} hs_drv hs_g{k} {hs_rg:.6g}",
            f"Vid{k} sw ls_d{k} 0",
            f"XLS{k} ls_d{k} ls_g{k} {ls_ks} ls_s {name}",
        ]
        if cfg.rg_on == cfg.rg_off:
            lines.append(f"Rgls{k} ls_drv ls_g{k} {cfg.rg_on:.6g}")
        else:  # separate source/sink paths, as on drivers with OUTH/OUTL pins
            lines.append(f"Rgls{k} ls_drv ls_g{k} R='V(ls_src,ls_kc) > {0.5*(on+off):.4g} ? "
                         f"{cfg.rg_on:.6g} : {cfg.rg_off:.6g}'")
        if n > 1:
            lines += [f"Rkhs{k} {hs_ks} hs_kc {cfg.r_ks:.6g}",
                      f"Rkls{k} {ls_ks} ls_kc {cfg.r_ks:.6g}"]
        if cfg.miller_clamp:
            lines.append(f"Sclamp{k} hs_g{k} hs_neg hs_kc hs_g{k} SCLAMP")
    if cfg.miller_clamp:
        lines.append(f".model SCLAMP SW(VT={-cfg.clamp_vth:.4g} VH=0.2 "
                     f"RON={cfg.clamp_r:.4g} ROFF=1e9)")
    body = "\n".join(lines)
    subtext = "\n".join(subs)
    return f"""* Double-pulse test: Vdc={cfg.vdc} V, I={cfg.i_test} A, Tj={cfg.tj} C, n_par={n}
.options temp=27 tnom=27 {solver} itl4=200
{subtext}
Vdc bus 0 {cfg.vdc:.6g}
{_power_loop(cfg)}
Lload bus sw {cfg.l_load:.6g}
Cload bus sw {cfg.c_load:.6g}
* high-side gate driver: held off (the clamp switch ties the gate to this rail)
Vhs hs_neg hs_kc {hs_off:.6g}
Rdhs hs_neg hs_drv {cfg.r_drv:.6g}
* low-side (DUT) gate driver: ideal source + output resistance
Vls ls_src ls_kc PWL({pwl})
{_driver_r(cfg)}
{body}
.tran 0.1n {t_stop:.6g} 0 {cfg.tmax:.6g}
.end
"""


def _power_loop(cfg: DPTConfig) -> str:
    """DC link -> (optional local HF capacitor) -> switches."""
    if cfg.c_hf <= 0:
        return "\n".join([f"Lp bus hs_d {cfg.l_loop/2:.6g}", f"Rpd bus hs_d {cfg.r_damp:.6g}",
                          f"Ln ls_s 0 {cfg.l_loop/2:.6g}", f"Rnd ls_s 0 {cfg.r_damp:.6g}"])
    return "\n".join([
        f"Lp bus bp {cfg.l_loop/2:.6g}", f"Rpd bus bp {cfg.r_damp:.6g}",
        f"Ln bn 0 {cfg.l_loop/2:.6g}", f"Rnd bn 0 {cfg.r_damp:.6g}",
        f"Chf bp chf1 {cfg.c_hf:.6g}", f"Rhf chf1 chf2 {cfg.esr_hf:.6g}", f"Lesl chf2 bn {cfg.esl_hf:.6g}",
        f"Lhp bp hs_d {cfg.l_hf/2:.6g}", f"Rhp bp hs_d {cfg.r_damp:.6g}",
        f"Lhn ls_s bn {cfg.l_hf/2:.6g}", f"Rhn ls_s bn {cfg.r_damp:.6g}",
    ])


def _driver_r(cfg: DPTConfig) -> str:
    r_on = cfg.r_drv if cfg.r_drv_on is None else cfg.r_drv_on
    r_off = cfg.r_drv if cfg.r_drv_off is None else cfg.r_drv_off
    if r_on == r_off:
        return f"Rdls ls_src ls_drv {r_on:.6g}"
    mid = 0.5 * (cfg.vgs_on + cfg.vgs_off)
    return f"Rdls ls_src ls_drv R='V(ls_src,ls_kc) > {mid:.4g} ? {r_on:.6g} : {r_off:.6g}'"


def _vectors(n: int) -> list[str]:
    return (["v(ls_d1,ls_kc)" if n == 1 else "v(ls_d1,ls_s)", "v(ls_g1,ls_kc)",
             "v(hs_g1,hs_kc)", "v(hs_d,sw)"] + [f"i(vid{k})" for k in range(1, n + 1)])


def simulate(dev: SiCMosfet, cfg: DPTConfig, keep=None) -> dict[str, np.ndarray]:
    vec = _vectors(cfg.n_par)
    for k, solver in enumerate(SOLVER_OPTIONS):
        try:
            raw = spice.run(netlist(dev, cfg, solver), vec, keep=keep)
            break
        except spice.SpiceError:
            if k == len(SOLVER_OPTIONS) - 1:
                raise
    ids = [raw[f"i(vid{k})"] for k in range(1, cfg.n_par + 1)]
    return {
        "t": raw["time"],
        "vds": raw[vec[0]],
        "id": np.sum(ids, axis=0),
        "id_each": ids,
        "vgs": raw[vec[1]],
        "vgs_hs": raw[vec[2]],
        "vds_hs": raw[vec[3]],
    }


def _first_cross(t, y, level, t_from, rising):
    """Time where y first crosses `level` after t_from (linear interpolation)."""
    m = t >= t_from
    tt, yy = t[m], y[m]
    s = (yy >= level) if rising else (yy <= level)
    idx = np.argmax(s)
    if not s[idx]:
        return np.nan
    if idx == 0:
        return tt[0]
    t_a, t_b, y_a, y_b = tt[idx - 1], tt[idx], yy[idx - 1], yy[idx]
    return t_a + (level - y_a) * (t_b - t_a) / (y_b - y_a)


def _energy(t, p, t_a, t_b):
    m = (t >= t_a) & (t <= t_b)
    return float(_trapz(p[m], t[m]))


def measure(w: dict, cfg: DPTConfig) -> dict:
    """Extract switching energies and timings from DPT waveforms."""
    t, vds, i_d, vgs = w["t"], w["vds"], w["id"], w["vgs"]
    p = vds * i_d
    swing = cfg.vgs_on - cfg.vgs_off
    g10, g90 = cfg.vgs_off + 0.1 * swing, cfg.vgs_off + 0.9 * swing
    t_off_cmd = cfg.t0 + _t1(cfg)
    t_on_cmd = t_off_cmd + cfg.t_gap

    # ---- turn-off (end of pulse 1)
    a = _first_cross(t, vgs, g90, t_off_cmd, rising=False)
    i_sw = float(np.interp(a, t, i_d))                     # actual current switched
    v90 = _first_cross(t, vds, 0.1 * cfg.vdc, a, rising=True)
    seg = (t > a) & (t < a + 0.5e-6)
    v_peak = float(vds[seg].max())
    i10 = _first_cross(t, i_d, 0.1 * i_sw, a, rising=False)
    i90 = _first_cross(t, i_d, 0.9 * i_sw, a, rising=False)
    b = _first_cross(t, i_d, 0.02 * i_sw, v90, rising=False)
    eoff = _energy(t, p, a, b)
    voff10 = _first_cross(t, vds, 0.1 * cfg.vdc, a, rising=True)
    voff90 = _first_cross(t, vds, 0.9 * cfg.vdc, a, rising=True)
    dv_off = _slope(t, vds, 0.2 * cfg.vdc, 0.8 * cfg.vdc, a, rising=True)
    di_off = _slope(t, i_d, 0.8 * i_sw, 0.2 * i_sw, a, rising=False)

    # ---- turn-on (start of pulse 2)
    c = _first_cross(t, vgs, g10, t_on_cmd, rising=True)
    # load current at the second turn-on = current switched off at the end of
    # pulse 1 (it barely decays during the short gap)
    i_load = i_sw
    ion10 = _first_cross(t, i_d, 0.1 * cfg.i_test, c, rising=True)
    ion90 = _first_cross(t, i_d, 0.9 * cfg.i_test, c, rising=True)
    d = _first_cross(t, vds, 0.02 * cfg.vdc, c, rising=False)
    eon = _energy(t, p, c, d)
    von90 = _first_cross(t, vds, 0.9 * cfg.vdc, c, rising=False)
    von10 = _first_cross(t, vds, 0.1 * cfg.vdc, c, rising=False)
    seg = (t > c) & (t < c + 0.5e-6)
    i_peak = float(i_d[seg].max())
    # freewheeling (high-side) device: its drain current is Id - I_load; it
    # recovers from the moment Id exceeds the load current. The load current
    # keeps ramping (Vdc / L) during pulse 2, so compare against that ramp.
    i_ref = i_load + cfg.vdc / cfg.l_load * np.clip(t - c, 0.0, None)
    x = i_d - i_ref
    t_rr = _first_cross(t, x, 0.0, c, rising=True)
    err = qrr = 0.0
    if np.isfinite(t_rr):
        t_pk = t[seg][np.argmax(x[seg])]
        t_end = _first_cross(t, x, 0.0, t_pk, rising=False)      # recovery current back to 0
        if np.isfinite(t_end) and t_end - t_rr < 300e-9:
            err = _energy(t, w["vds_hs"] * x, t_rr, t_end)
            qrr = _energy(t, x, t_rr, t_end)                     # incl. the HS Coss charge
    dv_on = _slope(t, vds, 0.8 * cfg.vdc, 0.2 * cfg.vdc, c, rising=False)
    di_on = _slope(t, i_d, 0.2 * cfg.i_test, 0.8 * cfg.i_test, c, rising=True)
    vhs = w["vgs_hs"][seg]

    return {
        "eon_mJ": eon * 1e3,
        "eoff_mJ": eoff * 1e3,
        "etot_mJ": (eon + eoff) * 1e3,
        "err_mJ": err * 1e3,
        "qrr_nC": qrr * 1e9,
        "i_off_A": i_sw,
        "v_peak_V": v_peak,
        "overshoot_V": v_peak - cfg.vdc,
        "dvdt_off_V_per_ns": _peak_slope(t, vds, a, a + 300e-9) * 1e-9,     # peak (2 ns avg)
        "dvdt_off_avg_V_per_ns": dv_off * 1e-9,                             # 20-80 % average
        "didt_off_A_per_ns": di_off * 1e-9,
        "dvdt_on_V_per_ns": _peak_slope(t, vds, c, c + 300e-9) * 1e-9,
        "dvdt_on_avg_V_per_ns": dv_on * 1e-9,
        "didt_on_A_per_ns": di_on * 1e-9,
        "i_rr_peak_A": i_peak - i_load,
        # switching times per the datasheet definition (Fig. 28): Vds edges
        "td_on_ns": (von90 - c) * 1e9,
        "tr_ns": (von10 - von90) * 1e9,
        "td_off_ns": (voff10 - a) * 1e9,
        "tf_ns": (voff90 - voff10) * 1e9,
        # current edges
        "t_ri_ns": (ion90 - ion10) * 1e9,
        "t_fi_ns": (i10 - i90) * 1e9,
        "hs_vgs_peak_V": float(vhs.max()),
        "v_peak_hs_V": float(w["vds_hs"][seg].max()),   # freewheeling device at LS turn-on
        "t_off_window": (a, b),
        "t_on_window": (c, d),
    }


def _peak_slope(t, y, t_a, t_b, window=2e-9):
    """Largest |dy/dt| between t_a and t_b after a `window` moving average."""
    tt = np.arange(t_a, t_b, 0.1e-9)
    yy = np.interp(tt, t, y)
    k = max(int(window / 0.1e-9), 1)
    ys = np.convolve(yy, np.ones(k) / k, mode="valid")
    return float(np.max(np.abs(np.diff(ys))) / 0.1e-9)


def _slope(t, y, y_a, y_b, t_from, rising):
    ta = _first_cross(t, y, y_a, t_from, rising)
    tb = _first_cross(t, y, y_b, t_from, rising)
    return abs((y_b - y_a) / (tb - ta))


def run(dev: SiCMosfet, cfg: DPTConfig, keep=None) -> tuple[dict, dict]:
    w = simulate(dev, cfg, keep=keep)
    return measure(w, cfg), w
