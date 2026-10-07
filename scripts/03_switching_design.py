"""
Step 3 - switching design studies, run with the calibrated model in the
inverter's own conditions (not the datasheet test fixture).

  A  gate-resistor selection, for two gate-drive options:
       A = UCC21750-Q1 drives the paralleled gates directly
       B = UCC21750-Q1 + a small totem-pole booster
  B  commutation-loop inductance budget
  C  switching-energy table (Id x Tj x Vdc) that feeds the loss model
  D  crosstalk: false turn-on of the idle device, and how -4 V + Miller clamp stop it
  E  all paralleled devices explicitly: current sharing with threshold spread
  F  short circuit + DESAT protection timing

Worst case for overshoot and dv/dt: highest bus voltage (850 V) and peak
phase current (300 A rms -> 424 A peak -> 106 A per device).

Studies A-D and F simulate ONE device with "per-device equivalent" values
(see sicinv/design.py); study E simulates every paralleled device explicitly, which
also checks that the per-device shortcut is fair.
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sicinv.device import SiCMosfet                                        # noqa: E402
from sicinv.dpt import DPTConfig, run                                      # noqa: E402
from sicinv.design import SPEC, CHOICE, per_device_loop, design_dpt, driver_r_per_device  # noqa: E402
from sicinv import shortcircuit as sc                                      # noqa: E402
from sicinv import plotstyle as ps                                         # noqa: E402
from sicinv.plotstyle import plt                                           # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
_trapz = getattr(np, "trapezoid", None) or np.trapz

dev = SiCMosfet.load(ps.DATA / "device_fit.json")
cal = json.loads((ps.DATA / "dpt_calibration.json").read_text())
N = CHOICE["n_par"]
I_PK_DEV = SPEC["i_peak_rms"] * np.sqrt(2) / N
E12 = [1.0, 1.5, 2.2, 3.3, 4.7, 6.8, 10.0, 15.0, 22.0]   # standard resistor values, ohm
DRIVER = {o: driver_r_per_device(o) for o in ("A", "B")}
out = {"i_peak_per_device_A": I_PK_DEV}


def design_cfg(rg_on, rg_off, option, **kw):
    return design_dpt(cal, rg_on, rg_off, option, **kw)


# ----------------------------------------------------------- A: gate resistors --
print("A  gate-resistor sweep ...", flush=True)
sweep = {}
for opt in ("A", "B"):
    rows = []
    for rg in E12:
        m, _ = run(dev, design_cfg(rg, rg, opt))
        rows.append({"rg": rg, **{k: m[k] for k in ("eon_mJ", "eoff_mJ", "err_mJ", "v_peak_V",
                                                     "v_peak_hs_V", "dvdt_on_V_per_ns",
                                                     "dvdt_off_V_per_ns", "i_rr_peak_A")}})
        print(f"   option {opt} Rg={rg:4.1f}  Eon {m['eon_mJ']:.2f}  Eoff {m['eoff_mJ']:.2f}  "
              f"Vpk {m['v_peak_V']:.0f}/{m['v_peak_hs_V']:.0f} V  dv/dt "
              f"{m['dvdt_on_V_per_ns']:.0f}/{m['dvdt_off_V_per_ns']:.0f} V/ns", flush=True)
    sweep[opt] = rows


def pick(rows, ok, key):
    good = [r for r in rows if ok(r)]
    return min(good, key=lambda r: r[key]) if good else max(rows, key=lambda r: r["rg"])


choice = {}
for opt, rows in sweep.items():
    # turn-off dv/dt at high current is set mostly by the load current charging
    # Coss (I / 2 Coss), not by the gate, so the dv/dt budget applies to turn-on;
    # Rg_off is set by the overshoot limit alone
    r_off = pick(rows, lambda r: r["v_peak_V"] <= SPEC["vpeak_limit"], "eoff_mJ")
    r_on = pick(rows, lambda r: r["v_peak_hs_V"] <= SPEC["vpeak_limit"]
                and r["dvdt_on_V_per_ns"] <= SPEC["dvdt_limit"], "eon_mJ")
    choice[opt] = {"rg_on": r_on["rg"], "rg_off": r_off["rg"],
                   "eon_mJ": r_on["eon_mJ"], "eoff_mJ": r_off["eoff_mJ"],
                   "etot_mJ": r_on["eon_mJ"] + r_off["eoff_mJ"]}
best = min(choice, key=lambda o: choice[o]["etot_mJ"])
RG_ON, RG_OFF = choice[best]["rg_on"], choice[best]["rg_off"]
out["gate_resistor_sweep"] = sweep
out["gate_resistor_choice"] = {"per_option": choice, "selected_option": best,
                               "rg_on_ohm": RG_ON, "rg_off_ohm": RG_OFF}
print(f"   -> option {best}: Rg_on = {RG_ON} ohm, Rg_off = {RG_OFF} ohm  {choice}", flush=True)


def cfg(**kw):
    return design_cfg(RG_ON, RG_OFF, best, **kw)


fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
for opt, col in (("A", ps.C2), ("B", ps.C1)):
    rows = sweep[opt]
    rg = [r["rg"] for r in rows]
    label = "driver only" if opt == "A" else "driver + booster"
    ax[0].plot(rg, [r["eon_mJ"] for r in rows], "-", color=col, label=f"Eon, {label}")
    ax[0].plot(rg, [r["eoff_mJ"] for r in rows], "--", color=col, label=f"Eoff, {label}")
    ax[1].plot(rg, [r["v_peak_V"] for r in rows], "-", color=col, label=f"turn-off, {label}")
    ax[1].plot(rg, [r["v_peak_hs_V"] for r in rows], "--", color=col, label=f"diode recovery, {label}")
    ax[2].plot(rg, [r["dvdt_off_V_per_ns"] for r in rows], "-", color=col, label=f"turn-off, {label}")
    ax[2].plot(rg, [r["dvdt_on_V_per_ns"] for r in rows], "--", color=col, label=f"turn-on, {label}")
ax[1].axhline(SPEC["vpeak_limit"], color=ps.INK_2, lw=1, ls=":")
ax[1].text(E12[-1], SPEC["vpeak_limit"], " limit", va="bottom", ha="right", color=ps.INK_2, fontsize=8)
ax[2].axhline(SPEC["dvdt_limit"], color=ps.INK_2, lw=1, ls=":")
ax[2].text(E12[-1], SPEC["dvdt_limit"], " limit", va="bottom", ha="right", color=ps.INK_2, fontsize=8)
for a, loc in zip(ax, ("upper left", "lower left", "upper right")):
    a.axvline(RG_ON, color=ps.GRID, lw=6, zorder=0)
    a.axvline(RG_OFF, color=ps.GRID, lw=6, zorder=0)
    a.set_xlabel("external gate resistor per device (Ω)")
    a.legend(fontsize=7.5, loc=loc)
ax[0].set(title="Switching energy per device", ylabel="mJ")
ax[1].set(title="Peak drain voltage", ylabel="V")
ax[2].set(title="Peak switching speed", ylabel="peak dv/dt (V/ns)")
fig.suptitle(f"Gate-resistor selection at 850 V / {I_PK_DEV:.0f} A per device: "
             f"Rg_on = {RG_ON} Ω, Rg_off = {RG_OFF} Ω (shaded), "
             f"{'driver + booster' if best == 'B' else 'driver only'}",
             x=0.01, ha="left", fontsize=12, fontweight="bold", color=ps.INK)
fig.tight_layout()
ps.save(fig, "03a_gate_resistor.png")

# ------------------------------------------------------- B: loop inductance --
print("B  loop-inductance budget ...", flush=True)
l_list = [0.5, 1, 2, 3, 4, 6]
vpk = []
for l_ext in l_list:
    m, _ = run(dev, cfg(l_hf=per_device_loop(l_ext, N)))
    vpk.append((m["v_peak_V"], m["v_peak_hs_V"]))
    print(f"   L_hf = {l_ext:4.1f} nH -> Vpeak {m['v_peak_V']:.0f} V / {m['v_peak_hs_V']:.0f} V", flush=True)
vmax = np.max(vpk, axis=1)
l_allowed = float(np.interp(SPEC["vpeak_limit"], vmax, l_list)) if vmax[-1] > SPEC["vpeak_limit"] else float(l_list[-1])
out["loop_inductance"] = {"l_hf_nH": l_list, "v_peak_V": [list(v) for v in vpk],
                          "max_allowed_l_hf_nH": l_allowed}
fig, a = plt.subplots(figsize=(6.4, 4.0))
a.plot(l_list, [v[0] for v in vpk], "o-", color=ps.C1, label="turn-off overshoot")
a.plot(l_list, [v[1] for v in vpk], "s--", color=ps.C2, label="diode-recovery overshoot")
a.axhline(SPEC["vpeak_limit"], color=ps.INK_2, lw=1, ls=":")
a.axvline(CHOICE["l_hf_nH"], color=ps.GRID, lw=6, zorder=0)
a.set(xlabel="loop inductance: local HF capacitors -> paralleled switches (nH)", ylabel="peak Vds (V)",
      title=f"Layout budget: L_hf ≤ {l_allowed:.1f} nH keeps Vds ≤ {SPEC['vpeak_limit']:.0f} V")
a.legend()
fig.tight_layout()
ps.save(fig, "03b_loop_inductance.png")

# --------------------------------------------------------- C: energy table --
print("C  switching-energy table ...", flush=True)
I_TAB = [5, 10, 20, 30, 40, 60, 80, 100, 120]
T_TAB = [25, 100, 175]
V_TAB = [600, 800]
E = {v: {k: np.zeros((len(T_TAB), len(I_TAB))) for k in ("eon", "eoff", "err")} for v in V_TAB}
for v in V_TAB:
    for a_, tj in enumerate(T_TAB):
        for b_, i in enumerate(I_TAB):
            m, _ = run(dev, cfg(vdc=float(v), tj=float(tj), i_test=float(i)))
            E[v]["eon"][a_, b_] = m["eon_mJ"] * 1e-3
            E[v]["eoff"][a_, b_] = m["eoff_mJ"] * 1e-3
            E[v]["err"][a_, b_] = max(m["err_mJ"], 0.0) * 1e-3
        print(f"   {v} V, {tj} C done", flush=True)


def kv(key):
    sel = np.array(I_TAB) >= 20
    r = E[800][key][:, sel] / np.maximum(E[600][key][:, sel], 1e-12)
    return float(np.mean(np.log(r) / np.log(800 / 600)))


table = {"i": I_TAB, "tj": T_TAB, "v_ref": 800.0,
         "eon": E[800]["eon"].tolist(), "eoff": E[800]["eoff"].tolist(), "err": E[800]["err"].tolist(),
         "kv_on": kv("eon"), "kv_off": kv("eoff"), "kv_rr": kv("err"),
         "meta": {"rg_on_ohm": RG_ON, "rg_off_ohm": RG_OFF, "driver_option": best,
                  "l_ext_nH": CHOICE["l_ext_nH"], "n_par": N,
                  "note": "per-device energies (J) from scripts/03_switching_design.py"}}
(RES / "sic_energy_table.json").write_text(json.dumps(table, indent=2) + "\n")
print(f"   kv_on {table['kv_on']:.2f}  kv_off {table['kv_off']:.2f}  kv_rr {table['kv_rr']:.2f}", flush=True)

fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
for a, key, title in zip(ax, ("eon", "eoff", "err"), ("Turn-on Eon", "Turn-off Eoff", "Diode recovery Err")):
    for k, (tj, col) in enumerate(zip(T_TAB, (ps.C1, ps.C3, ps.C2))):
        a.plot(I_TAB, E[800][key][k] * 1e3, "o-", ms=4, color=col, label=f"{tj} °C")
    a.set(title=title, xlabel="current per device (A)")
    a.legend()
ax[0].set_ylabel("energy per device (mJ), 800 V")
fig.suptitle("Switching-energy table used by the loss model (design Rg, layout and driver)",
             x=0.01, ha="left", fontsize=12, fontweight="bold", color=ps.INK)
fig.tight_layout()
ps.save(fig, "03c_energy_table.png")

# ------------------------------------------------------------- D: crosstalk --
print("D  crosstalk / Miller clamp ...", flush=True)
cases = [("0 V off, no clamp", dict(hs_vgs_off=0.0, miller_clamp=False), ps.C2),
         ("-4 V off, no clamp", dict(miller_clamp=False), ps.C4),
         ("-4 V off + Miller clamp (design)", dict(), ps.C1)]
fig, a = plt.subplots(figsize=(7.2, 4.2))
xt = {}
for label, kw, col in cases:
    m, w = run(dev, cfg(tj=175.0, **kw))
    t0 = m["t_on_window"][0]
    sel = (w["t"] > t0 - 20e-9) & (w["t"] < t0 + 150e-9)
    a.plot((w["t"][sel] - t0) * 1e9, w["vgs_hs"][sel], color=col, label=label)
    xt[label] = {"hs_vgs_peak_V": m["hs_vgs_peak_V"], "eon_mJ": m["eon_mJ"]}
    print(f"   {label:34s} HS Vgs peak {m['hs_vgs_peak_V']:+.2f} V  Eon {m['eon_mJ']:.2f} mJ", flush=True)
vth175 = dev.vt(175.0)
a.axhline(vth175, color=ps.INK_2, lw=1, ls=":")
a.text(148, vth175, f"threshold at 175 °C ({vth175:.1f} V)", ha="right", va="bottom", color=ps.INK_2, fontsize=8)
a.set(xlabel="time from low-side turn-on (ns)", ylabel="gate voltage of the OFF device (V)",
      title="Crosstalk: dv/dt pushes the idle gate up through Cgd")
a.legend(fontsize=8)
fig.tight_layout()
ps.save(fig, "03d_crosstalk.png")
out["crosstalk_175C"] = xt

# ----------------------------------------------------------- E: paralleling --
print(f"E  {N} devices in parallel ...", flush=True)
r_on_shared = CHOICE["booster_r"] if best == "B" else CHOICE["drv_routh"]
r_off_shared = CHOICE["booster_r"] if best == "B" else CHOICE["drv_routl"]
par_base = DPTConfig(
    vdc=SPEC["vdc_max"], i_test=I_PK_DEV * N, tj=25.0, n_par=N,
    rg_on=RG_ON, rg_off=RG_OFF, r_drv=1e-3, r_drv_on=r_on_shared, r_drv_off=r_off_shared,
    vgs_on=CHOICE["vgs_on"], vgs_off=CHOICE["vgs_off"],
    l_loop=CHOICE["l_ext_nH"] * 1e-9, r_damp=CHOICE["r_damp_ohm"], l_load=10e-6, c_load=200e-12,
    c_hf=CHOICE["c_hf_uF"] * 1e-6, l_hf=CHOICE["l_hf_nH"] * 1e-9,
    esl_hf=CHOICE["esl_hf_nH"] * 1e-9, esr_hf=CHOICE["esr_hf_mOhm"] * 1e-3,
    edge=cal["edge_ns"] * 1e-9, t_gap=1.0e-6, tmax=0.5e-9, miller_clamp=True, clamp_vth=-2.0, clamp_r=1.0,
    hs_rg=RG_OFF + r_off_shared)
spread = tuple(np.round(np.linspace(0.3, -0.3, N), 3))   # +/-0.3 V threshold spread
par = {}
fig, ax = plt.subplots(1, 2, figsize=(12, 4.0))
for tag, offs in (("matched", ()), ("threshold spread +-0.3 V", spread)):
    m, w = run(dev, par_base.with_(vth_offsets=offs))
    t = w["t"]
    shares = []
    for (ta, tb) in (m["t_on_window"], m["t_off_window"]):
        sel = (t >= ta) & (t <= tb)
        e_k = [float(_trapz((w["vds"] * ik)[sel], t[sel])) for ik in w["id_each"]]
        shares.append([e / sum(e_k) for e in e_k])
    par[tag] = {"etot_mJ_total": m["etot_mJ"], "eon_share": shares[0], "eoff_share": shares[1],
                "peak_current_each_A": [float(np.max(ik)) for ik in w["id_each"]]}
    print(f"   {tag:24s} Eon shares {np.round(shares[0], 3)}  Eoff shares {np.round(shares[1], 3)}", flush=True)
    if offs:
        for k, (ik, col) in enumerate(zip(w["id_each"], ps.SERIES)):
            for a, (ta, _) in zip(ax, (m["t_on_window"], m["t_off_window"])):
                sel = (t > ta - 20e-9) & (t < ta + 160e-9)
                a.plot((t[sel] - ta) * 1e9, ik[sel], color=col,
                       label=f"device {k+1} (Vth {offs[k]:+.1f} V)")
ax[0].set(title="Turn-on: low-Vth device turns on first and takes more current",
          xlabel="time (ns)", ylabel="drain current (A)")
ax[1].set(title="Turn-off: low-Vth device turns off last", xlabel="time (ns)", ylabel="drain current (A)")
ax[0].legend(fontsize=8)
fig.tight_layout()
ps.save(fig, "03e_paralleling.png")
out["paralleling"] = par

# ---------------------------------------------------------- F: short circuit --
print("F  short circuit + DESAT ...", flush=True)
c_blk_list = [10e-12, 22e-12, 47e-12]
timing = {f"{c*1e12:.0f}pF": {"typ_us": sc.desat_delay(c) * 1e6,
                               "worst_us": sc.desat_delay(c, sc.DESAT_WORST) * 1e6} for c in c_blk_list}
C_BLK = 10e-12           # keeps worst-case fault clearing under ~2 us
r_on_dev = RG_ON + DRIVER[best][0]
sc_base = sc.SCConfig(vdc=SPEC["vdc_max"], tj=25.0,
                      l_sc=per_device_loop(CHOICE["l_hf_nH"] + CHOICE["esl_hf_nH"], N),
                      r_damp=N * CHOICE["r_damp_ohm"], rg_on=r_on_dev,
                      vgs_on=CHOICE["vgs_on"], vgs_off=CHOICE["vgs_off"], edge=cal["edge_ns"] * 1e-9)
runs = {}
fig, ax = plt.subplots(1, 2, figsize=(12, 4.0))
for tag, p, col in (("typical", sc.DESAT_TYP, ps.C1), ("worst case", sc.DESAT_WORST, ps.C2)):
    c = sc_base.with_(t_trip=sc.desat_delay(C_BLK, p), i_sto=p["i_sto"] / N)
    m, w = sc.run(dev, c)
    runs[tag] = m
    print(f"   {tag:10s} trip {m['t_trip_us']:.2f} us  Ipk {m['i_peak_A']:.0f} A  "
          f"fault over after {m['t_fault_total_us']:.2f} us  E {m['e_sc_mJ']:.0f} mJ  "
          f"Vpk {m['v_peak_off_V']:.0f} V", flush=True)
    tt = (w["t"] - c.t_on) * 1e6
    sel = tt > -0.1
    ax[0].plot(tt[sel], w["id"][sel], color=col, label=f"{tag}: off after {m['t_fault_total_us']:.1f} µs")
    ax[1].plot(tt[sel], w["vds"][sel], color=col, label=tag)
ax[0].axvline(2.0, color=ps.INK_2, lw=1, ls=":")
ax[0].text(2.02, ax[0].get_ylim()[1] * 0.9, "~2 µs: typical SiC\nwithstand time", fontsize=8, color=ps.INK_2)
ax[0].set(title="Shoot-through current per device (upper bound)",
          xlabel="time from gate turn-on (µs)", ylabel="A")
ax[1].set(title="Drain voltage during the soft turn-off", xlabel="time from gate turn-on (µs)", ylabel="V")
for a in ax:
    a.legend(fontsize=8)
fig.tight_layout()
ps.save(fig, "03f_short_circuit.png")
out["short_circuit"] = {"c_blk_pF": C_BLK * 1e12, "desat_delay_options": timing, "runs": runs}

(RES / "switching_design.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
print("done", flush=True)
