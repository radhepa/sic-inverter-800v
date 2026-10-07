"""
Compact model of a 1200 V / 21 mOhm SiC MOSFET (calibrated to the C3M0021120K
datasheet), usable two ways:

  1. In Python, for loss calculations and for fitting (`SiCMosfet.id_channel`,
     `SiCMosfet.cgd`, ... are plain numpy functions).
  2. In ngspice, through `SiCMosfet.subckt(Tj)`, which writes the *same*
     equations as a SPICE subcircuit.

Equivalent circuit of one die (subcircuit pins: d, g, ks = Kelvin source,
s = power source)

    element   nodes        what it models
    Ld        d  - d1      drain lead / bond-wire inductance
    Rd(T)     d1 - di      drift-region resistance (unipolar current)
    B_ch      di - si      channel current Id(Vgs, Vds), both directions
    Cds(V)    d1 - si      drain-source (body-junction) depletion capacitance
    Cgd(V)    di - gi      gate-drain ("Miller") capacitance
    Cgs       gi - si      gate-source capacitance (oxide, constant)
    D_body    si - d1      body PiN diode (anode at source); it bypasses Rd
                           because in bipolar conduction the drift region is
                           flooded with carriers (conductivity modulation)
    Lg, Rg_int g - gi      gate lead inductance + internal gate resistance
    Lks, Ls   ks/s - si    Kelvin-source and power-source lead inductances

Physics captured
  * Channel: square-law MOSFET with a soft (softplus) turn-on, threshold and
    transconductance interpolated between the 25 C and 175 C transfer curves.
    Works in both directions (3rd-quadrant "synchronous rectification").
    An empirical threshold shift at high Vds reproduces the datasheet's Miller
    plateau; without it the simulated turn-on is ~30 % too fast.
  * Drift region: series resistance with a T^n temperature law (n ~ 2.5 for
    SiC electron mobility), which is why Rds(on) rises ~1.7x from 25 to 175 C.
  * Capacitances: Cgs constant; Cgd and Cds voltage-dependent, including the
    sharp drop of Crss near 10 V when the JFET region pinches off, and the
    large gate-drain oxide capacitance when the gate sits above the drain.
  * Body diode: PiN diode with a high (~3 V) knee and a short carrier
    lifetime (reverse recovery) that grows with temperature.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
import json
from pathlib import Path

import numpy as np

K_BOLTZ_Q = 8.617333e-5  # Boltzmann constant / electron charge, V/K
SPICE_TEMP_C = 27.0      # every netlist runs at this temperature; Tj is
                         # applied through the parameters instead

_trapz = getattr(np, "trapezoid", None) or np.trapz


def _lerp(t_c: float, v25: float, v175: float) -> float:
    """Linear interpolation (and extrapolation) in junction temperature."""
    return v25 + (v175 - v25) * (t_c - 25.0) / 150.0


@dataclass
class SiCMosfet:
    # --- channel (fit to datasheet Fig. 7, transfer characteristic) ---------
    vt25: float = 2.94        # effective threshold at 25 C, V
    vt175: float = 2.42       # effective threshold at 175 C, V
    k25: float = 7.84         # transconductance parameter at 25 C, A/V^2
    k175: float = 8.42        # transconductance parameter at 175 C, A/V^2
    s_on: float = 0.15        # softness of turn-on (softplus width), V
    lam: float = 0.0          # channel-length modulation, 1/V
    # empirical high-voltage threshold shift: the datasheet Miller plateau
    # (gate charge at 800 V, Fig. 12) sits ~1 V above what the Vds = 20 V
    # transfer curve predicts. Vt_eff = Vt + dvt_hv * Vds / (Vds + v_hv)
    dvt_hv: float = 0.0       # V
    v_hv: float = 50.0        # V

    # --- drift region (fit to Rds(on) vs T, Fig. 4) --------------------------
    rd25: float = 0.0104      # ohm at 25 C
    rd_exp: float = 2.5       # Rd ~ (T/298 K)^rd_exp

    # --- capacitances (fit to Fig. 16-18 and gate charge Fig. 12) ------------
    cgs: float = 4.80e-9
    # Cgd(v), v = V(drain) - V(gate)
    ca: float = 150e-12       # depletion term amplitude
    va: float = 4.0
    ma: float = 0.55
    cp: float = 400e-12       # JFET-pinch term amplitude
    vp: float = 10.0          # pinch-off voltage
    wp: float = 1.0           # pinch transition width
    cacc: float = 3.0e-9      # Cgd when the gate is well above the drain
    vacc: float = 2.0         # how fast Cgd reaches cacc for v < 0
    # Cds(v), v = V(drain) - V(source)
    cb: float = 3.0e-9
    vb: float = 1.0
    mb: float = 0.55
    cinf: float = 150e-12     # high-voltage floor (drift fully depleted)
    cq: float = 300e-12       # JFET-pinch term amplitude (shares vp, wp)

    # --- body diode (fit to Fig. 9/10 and Vsd table) -------------------------
    dio_is25: float = 1.0e-6
    dio_is175: float = 8.5e-4
    dio_n25: float = 9.7     # ideality factor (high-injection PiN diode)
    dio_n175: float = 9.7
    dio_rs: float = 0.004
    dio_tt25: float = 6e-9
    dio_tt175: float = 15e-9

    # --- gate + package (datasheet Rg_int; typical TO-247-4 inductances) -----
    rg_int: float = 3.3
    l_d: float = 3e-9
    l_s: float = 3e-9
    l_ks: float = 3e-9
    l_g: float = 3e-9

    name: str = "SIC1200V21M"
    notes: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ io --
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "SiCMosfet":
        return cls(**json.loads(Path(path).read_text()))

    # --------------------------------------------- temperature dependence --
    def vt(self, tj: float) -> float:
        return _lerp(tj, self.vt25, self.vt175)

    def k(self, tj: float) -> float:
        return _lerp(tj, self.k25, self.k175)

    def rd(self, tj: float) -> float:
        return self.rd25 * ((tj + 273.15) / 298.15) ** self.rd_exp

    def dio_is(self, tj: float) -> float:
        return float(np.exp(_lerp(tj, np.log(self.dio_is25), np.log(self.dio_is175))))

    def dio_n(self, tj: float) -> float:
        return _lerp(tj, self.dio_n25, self.dio_n175)

    def dio_tt(self, tj: float) -> float:
        return _lerp(tj, self.dio_tt25, self.dio_tt175)

    # ------------------------------------------------------------ channel --
    def _vov(self, vg_eff, tj, v_blk=0.0):
        """Smooth overdrive voltage: ~0 below threshold, Vgs - Vt above it.
        v_blk is the blocking voltage across the channel (drives the shift)."""
        v_blk = np.maximum(np.asarray(v_blk, float), 0.0)
        vt = self.vt(tj) + self.dvt_hv * v_blk / (v_blk + self.v_hv)
        x = (np.asarray(vg_eff, float) - vt) / self.s_on
        return self.s_on * np.logaddexp(0.0, x)

    def id_channel(self, vgs, vds, tj: float = 25.0):
        """Channel current (A) for the *internal* Vgs and Vds (no Rd)."""
        vgs = np.asarray(vgs, float)
        vds = np.asarray(vds, float)
        fwd = vds >= 0
        # forward: gate drive is Vgs; reverse: the "source" is really the drain
        vov = np.where(fwd, self._vov(vgs, tj, vds), self._vov(vgs - vds, tj, -vds))
        x = np.minimum(np.abs(vds), vov)
        mag = self.k(tj) * (vov * x - 0.5 * x * x) * (1.0 + self.lam * np.abs(vds))
        return np.where(fwd, mag, -mag)

    def vds_on(self, i_d: float, tj: float, vgs: float = 15.0) -> float:
        """Terminal Vds for a forward current i_d (channel + drift)."""
        lo, hi = 0.0, 50.0
        for _ in range(80):  # bisection on the channel voltage
            mid = 0.5 * (lo + hi)
            if self.id_channel(vgs, mid, tj) < i_d:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi) + i_d * self.rd(tj)

    def rds_on(self, tj: float, i_d: float = 50.0, vgs: float = 15.0) -> float:
        return self.vds_on(i_d, tj, vgs) / i_d

    # ------------------------------------------------------- capacitances --
    def _pinch(self, v):
        return 0.5 * (1.0 - np.tanh((np.asarray(v, float) - self.vp) / self.wp))

    def _cgd_pos(self, v):
        v = np.maximum(np.asarray(v, float), 0.0)
        return self.ca * (1.0 + v / self.va) ** (-self.ma) + self.cp * self._pinch(v)

    def cgd(self, v):
        """Gate-drain capacitance vs v = Vd - Vg (F)."""
        v = np.asarray(v, float)
        c0 = self._cgd_pos(0.0)
        neg = c0 + (self.cacc - c0) * (1.0 - np.exp(np.minimum(v, 0.0) / self.vacc))
        return np.where(v >= 0, self._cgd_pos(v), neg)

    def cds(self, v):
        """Drain-source capacitance vs v = Vds (F); flat below 0 V."""
        v = np.maximum(np.asarray(v, float), 0.0)
        return (self.cb * (1.0 + v / self.vb) ** (-self.mb) + self.cinf
                + self.cq * self._pinch(v))

    # datasheet-style small-signal capacitances (Vgs = 0, so Vdg = Vds)
    def crss(self, vds):
        return self.cgd(vds)

    def coss(self, vds):
        return self.cgd(vds) + self.cds(vds)

    def ciss(self, vds):
        return self.cgs + self.cgd(vds)

    def eoss(self, vds: float, n: int = 4000) -> float:
        """Energy stored in Coss at vds (J): integral of v*Coss(v) dv."""
        v = np.linspace(0.0, vds, n)
        return float(_trapz(v * self.coss(v), v))

    def qoss(self, vds: float, n: int = 4000) -> float:
        v = np.linspace(0.0, vds, n)
        return float(_trapz(self.coss(v), v))

    # ------------------------------------------------------ gate charge --
    def vgs_for_current(self, i_d, vds_ch, tj: float):
        """Gate voltage needed to carry i_d at internal channel voltage vds_ch
        (vectorised bisection; i_d and vds_ch may be arrays)."""
        i_d, vds_ch = np.broadcast_arrays(np.asarray(i_d, float), np.asarray(vds_ch, float))
        lo, hi = np.full(i_d.shape, -5.0), np.full(i_d.shape, 40.0)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            low = self.id_channel(mid, vds_ch, tj) < i_d
            lo, hi = np.where(low, mid, lo), np.where(low, hi, mid)
        out = 0.5 * (lo + hi)
        return float(out) if out.ndim == 0 else out

    def gate_charge(self, vdd: float = 800.0, i_load: float = 20.0, tj: float = 25.0,
                    vg_off: float = -4.0, vg_on: float = 15.0):
        """Quasi-static gate-charge curve for an inductive-clamped turn-on.

        Returns (Q [C], Vgs [V]). Phase 1: the gate charges Cgs + Cgd while Vds
        stays at vdd (the freewheeling diode still carries the load). Phase 2
        (Miller plateau): Vds collapses; the gate voltage is whatever the
        channel needs to keep carrying i_load at that Vds.
        """
        v_pl = self.vgs_for_current(i_load, vdd, tj)
        # phase 1
        vg1 = np.linspace(vg_off, v_pl, 400)
        vd1 = np.full_like(vg1, vdd)
        # phase 2: walk Vds down until the gate reaches vg_on
        vds_end = self.vds_on(i_load, tj, vg_on) - i_load * self.rd(tj)
        vd2 = np.geomspace(vdd, vds_end, 1500)
        vg2 = self.vgs_for_current(i_load, vd2, tj)
        vg = np.concatenate([vg1, vg2[1:]])
        vd = np.concatenate([vd1, vd2[1:]])
        dvg, dvd = np.diff(vg), np.diff(vd)
        vmid_dg = 0.5 * ((vd - vg)[1:] + (vd - vg)[:-1])
        dq = self.cgs * dvg + self.cgd(vmid_dg) * (dvg - dvd)
        return np.concatenate([[0.0], np.cumsum(dq)]), vg

    # ------------------------------------------------------- body diode --
    def vsd(self, i_f, tj: float) -> np.ndarray:
        """Body-diode forward voltage with the channel off (Vgs = -4 V)."""
        i_f = np.asarray(i_f, float)
        vt = K_BOLTZ_Q * (tj + 273.15)
        return (self.dio_n(tj) * vt * np.log(i_f / self.dio_is(tj) + 1.0)
                + i_f * self.dio_rs)

    # ------------------------------------------------------------ SPICE --
    def _spice_cgd(self, v: str) -> str:
        c0 = float(self._cgd_pos(0.0))
        pos = (f"{self.ca:.6g}*pow(1+({v})/{self.va:.6g},{-self.ma:.6g})"
               f"+{self.cp:.6g}*0.5*(1-tanh((({v})-{self.vp:.6g})/{self.wp:.6g}))")
        neg = f"{c0:.6g}+{self.cacc - c0:.6g}*(1-exp(({v})/{self.vacc:.6g}))"
        return f"({v}) >= 0 ? {pos} : {neg}"

    def _spice_cds(self, v: str) -> str:
        vv = f"max({v},0)"
        return (f"{self.cb:.6g}*pow(1+{vv}/{self.vb:.6g},{-self.mb:.6g})+{self.cinf:.6g}"
                f"+{self.cq:.6g}*0.5*(1-tanh(({vv}-{self.vp:.6g})/{self.wp:.6g}))")

    def _spice_channel(self, tj: float) -> str:
        vt, k, s = self.vt(tj), self.k(tj), self.s_on

        def vov(vg: str, vblk: str) -> str:
            vb = f"max({vblk},0)"
            vte = f"({vt:.6g}+{self.dvt_hv:.6g}*{vb}/({vb}+{self.v_hv:.6g}))"
            return f"{s:.6g}*ln(1+exp((({vg})-{vte})/{s:.6g}))"

        def mag(vov_expr: str, vabs: str) -> str:
            x = f"min({vabs},{vov_expr})"
            return f"{k:.6g}*(1+{self.lam:.6g}*{vabs})*(({vov_expr})*{x}-0.5*{x}*{x})"

        fwd = mag(vov("V(gi,si)", "V(di,si)"), "V(di,si)")
        rev = mag(vov("V(gi,si)-V(di,si)", "-V(di,si)"), "(-V(di,si))")
        return f"V(di,si) >= 0 ? {fwd} : -({rev})"

    def _spice_dio_n(self, tj: float) -> float:
        # ngspice evaluates N*kT/q at SPICE_TEMP_C, so scale N to give the
        # thermal voltage of the real junction temperature
        return self.dio_n(tj) * (tj + 273.15) / (SPICE_TEMP_C + 273.15)

    def subckt(self, tj: float = 25.0, name: str | None = None) -> str:
        """SPICE subcircuit for this device at junction temperature tj (C)."""
        name = name or f"{self.name}_T{int(round(tj))}"
        return "\n".join([
            f"* {self.name} SiC MOSFET model at Tj = {tj:g} C",
            f".subckt {name} d g ks s",
            f"Ld d d1 {self.l_d:.4g}",
            f"Rd d1 di {self.rd(tj):.6g}",
            f"Lg g g1 {self.l_g:.4g}",
            f"Rgint g1 gi {self.rg_int:.4g}",
            f"Lks ks si {self.l_ks:.4g}",
            f"Ls s si {self.l_s:.4g}",
            f"Cgs gi si {self.cgs:.6g}",
            f"Cgd di gi C='{self._spice_cgd('V(di,gi)')}'",
            f"Cds d1 si C='{self._spice_cds('V(d1,si)')}'",
            f"Bch di si I={self._spice_channel(tj)}",
            "Dbody si d1 DBODY",
            f".model DBODY D(IS={self.dio_is(tj):.6g} N={self._spice_dio_n(tj):.6g} "
            f"RS={self.dio_rs:.6g} TT={self.dio_tt(tj):.6g} CJO=0 BV=1500)",
            f".ends {name}",
            "",
        ])
