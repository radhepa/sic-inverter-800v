"""
Step 2 - calibrate the dynamic behaviour against the datasheet's double-pulse
results, then validate on data that was NOT used for calibration.

The static fit (step 1) fixes the device itself. What the datasheet doesn't
tell us is the test fixture and the diode's carrier lifetime:

    r_drv   gate-driver output resistance          edge    driver edge time
    L       power-loop inductance                  C_L     load-inductor winding capacitance
    tau     body-diode carrier lifetime at 175 C   (25 C value = tau/3: the
            body-diode share of Eon in Fig. 26 grows ~3x from 25 to 175 C)

They are fitted to the datasheet's ONE standard operating point
(800 V, 50 A, Rg = 2.5 ohm) at 25 C (Eon, Eoff, 4 switching times) and 175 C
(Eon, Qrr), then checked against 13 other datasheet points:

    validation : Id sweep 10-80 A (Fig. 24), Rg sweep 2.5-20 ohm (Fig. 25),
                 Tj sweep 25-175 C (Fig. 26)

Output: data/dpt_calibration.json, data/dpt_validation.csv,
        figures/02_dpt_validation.png, figures/02_dpt_waveforms.png
Run with --plots to redraw the figures from the saved results only.
"""
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sicinv.device import SiCMosfet                       # noqa: E402
from sicinv.dpt import DPTConfig, run                     # noqa: E402
from sicinv import plotstyle as ps                        # noqa: E402
from sicinv.plotstyle import plt                          # noqa: E402

ds = json.loads((ps.DATA / "c3m0021120k_datasheet.json").read_text(encoding="utf-8"))
sc = ds["scalars"]
dev0 = SiCMosfet.load(ps.DATA / "device_fit.json")
# The datasheet used a 157 uH load inductor; 40 uH only shortens the first
# pulse (4x faster runs) and changes the switching energies by < 1 %.
base = DPTConfig(vdc=800, i_test=50, rg_on=2.5, rg_off=2.5, l_load=40e-6, tmax=0.5e-9, t_gap=1e-6)

TARGET_25 = {"eon_mJ": 1.02, "eoff_mJ": 0.30,
             "td_on_ns": sc["td_on_ns"], "tr_ns": sc["tr_ns"],
             "td_off_ns": sc["td_off_ns"], "tf_ns": sc["tf_ns"]}
TARGET_175 = {"eon_mJ": sc["Eon_mJ_body_diode_175C"], "qrr_nC": sc["Qrr_nC_175C"]}
WEIGHT = {"eon_mJ": 1.0, "eoff_mJ": 1.0, "td_on_ns": 0.3, "tr_ns": 0.3,
          "td_off_ns": 0.3, "tf_ns": 0.3, "qrr_nC": 0.5}
PARAMS = ["r_drv", "L_nH", "edge_ns", "C_L_pF", "tau175_ns"]


def unpack(p):
    r_drv, l_nh, edge_ns, cl_pf, tau175 = p
    dev = replace(dev0, dio_tt25=tau175 / 3 * 1e-9, dio_tt175=tau175 * 1e-9)
    cfg = base.with_(r_drv=r_drv, l_loop=l_nh * 1e-9, edge=edge_ns * 1e-9, c_load=cl_pf * 1e-12)
    return dev, cfg


def residuals(p):
    dev, cfg = unpack(p)
    m25, _ = run(dev, cfg.with_(tj=25.0))
    m175, _ = run(dev, cfg.with_(tj=175.0))
    r = [WEIGHT[k] * (m25[k] / v - 1.0) for k, v in TARGET_25.items()]
    r += [WEIGHT[k] * (m175[k] / v - 1.0) for k, v in TARGET_175.items()]
    print("  " + "  ".join(f"{n}={x:.2f}" for n, x in zip(PARAMS, p))
          + "  ->  25C: " + " ".join(f"{k.split('_')[0]} {m25[k]:.3g}" for k in TARGET_25)
          + "  | 175C: " + " ".join(f"{k.split('_')[0]} {m175[k]:.3g}" for k in TARGET_175),
          flush=True)
    return r


if "--plots" not in sys.argv:
    print("calibrating at 800 V / 50 A / 2.5 ohm (25 C and 175 C) ...")
    fit = least_squares(residuals, x0=[3.0, 40.0, 8.0, 30.0, 15.0],
                        bounds=([0.2, 5.0, 1.0, 1.0, 1.0], [10.0, 100.0, 40.0, 300.0, 100.0]),
                        diff_step=0.05, xtol=2e-3, ftol=2e-3, max_nfev=40)
    dev, cfg = unpack(fit.x)
    dev.notes = {**dev.notes, "dynamic": "diode lifetimes from scripts/02_calibrate_dpt.py"}
    dev.save(ps.DATA / "device_fit.json")
    m25, _ = run(dev, cfg.with_(tj=25.0))
    m175, _ = run(dev, cfg.with_(tj=175.0))
    calib = {"r_drv_ohm": cfg.r_drv, "l_loop_nH": cfg.l_loop * 1e9, "edge_ns": cfg.edge * 1e9,
             "c_load_pF": cfg.c_load * 1e12, "dio_tt25_ns": dev.dio_tt25 * 1e9,
             "dio_tt175_ns": dev.dio_tt175 * 1e9, "r_damp_ohm": cfg.r_damp,
             "fit_25C": {k: m25[k] for k in TARGET_25}, "target_25C": TARGET_25,
             "fit_175C": {k: m175[k] for k in TARGET_175}, "target_175C": TARGET_175}
    (ps.DATA / "dpt_calibration.json").write_text(json.dumps(calib, indent=2) + "\n")
    print("calibration:", json.dumps(calib, indent=2))
else:
    calib = json.loads((ps.DATA / "dpt_calibration.json").read_text())
    dev = dev0
    cfg = base.with_(r_drv=calib["r_drv_ohm"], l_loop=calib["l_loop_nH"] * 1e-9,
                     edge=calib["edge_ns"] * 1e-9, c_load=calib["c_load_pF"] * 1e-12)

# ------------------------------------------------------------- validation --
rows = []


def check(fig, x_name, x, cfg_i, eon_ref, eoff_ref, used_for_cal=False):
    m, _ = run(dev, cfg_i)
    rows.append({"figure": fig, "x_name": x_name, "x": x,
                 "eon_model_mJ": m["eon_mJ"], "eon_datasheet_mJ": eon_ref,
                 "eoff_model_mJ": m["eoff_mJ"], "eoff_datasheet_mJ": eoff_ref,
                 "calibration_point": used_for_cal})
    print(f"  {fig:7s} {x_name}={x:<6g} Eon {m['eon_mJ']:.3f} (ds {eon_ref:.2f})  "
          f"Eoff {m['eoff_mJ']:.3f} (ds {eoff_ref:.2f})")


if "--plots" not in sys.argv:
    print("validating ...")
    f24 = ds["eon_eoff_vs_id_800V_fig24"]
    for (i, eon), (_, eoff) in zip(f24["Eon"], f24["Eoff"]):
        check("Fig. 24", "Id_A", i, cfg.with_(tj=25.0, i_test=i), eon, eoff,
              used_for_cal=(i == 50))           # 50 A, 2.5 ohm, 25 C IS the calibration point
    f25 = ds["eon_eoff_vs_rg_fig25"]
    for (rg, eon), (_, eoff) in zip(f25["Eon"], f25["Eoff"]):
        check("Fig. 25", "Rg_ohm", rg, cfg.with_(tj=25.0, rg_on=rg, rg_off=rg), eon, eoff,
              used_for_cal=(rg == 2.5))
    f26 = ds["eon_eoff_vs_tj_fig26"]
    for (tj, eon), (_, eoff) in zip(f26["Eon"], f26["Eoff"]):
        check("Fig. 26", "Tj_C", tj, cfg.with_(tj=float(tj)), eon, eoff,
              used_for_cal=(tj in (25, 175)))

    with open(ps.DATA / "dpt_validation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
else:
    with open(ps.DATA / "dpt_validation.csv") as f:
        rows = [{**r, "x": float(r["x"]), "calibration_point": r["calibration_point"] == "True",
                 **{k: float(r[k]) for k in r if k.endswith("_mJ")}} for r in csv.DictReader(f)]

val = [r for r in rows if not r["calibration_point"]]
err_on = np.array([r["eon_model_mJ"] / r["eon_datasheet_mJ"] - 1 for r in val])
err_tot = np.array([(r["eon_model_mJ"] + r["eoff_model_mJ"])
                    / (r["eon_datasheet_mJ"] + r["eoff_datasheet_mJ"]) - 1 for r in val])
summary = {"n_validation_points": len(val),
           "eon_mean_abs_err_pct": 100 * float(np.mean(np.abs(err_on))),
           "etot_mean_abs_err_pct": 100 * float(np.mean(np.abs(err_tot))),
           "etot_max_abs_err_pct": 100 * float(np.max(np.abs(err_tot)))}
calib["validation"] = summary
(ps.DATA / "dpt_calibration.json").write_text(json.dumps(calib, indent=2) + "\n")
print("validation summary:", summary)

# ---------------------------------------------------------------- figures --
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))
for a, figname, xlab, title in ((ax[0], "Fig. 24", "Drain current (A)", "vs drain current"),
                                (ax[1], "Fig. 25", "External gate resistor Rg (Ω)", "vs gate resistor"),
                                (ax[2], "Fig. 26", "Junction temperature (°C)", "vs temperature")):
    rr = [r for r in rows if r["figure"] == figname]
    x = [r["x"] for r in rr]
    a.plot(x, [r["eon_model_mJ"] for r in rr], color=ps.C1, label="Eon model")
    a.plot(x, [r["eoff_model_mJ"] for r in rr], color=ps.C2, label="Eoff model")
    ps.ref_points(a, x, [r["eon_datasheet_mJ"] for r in rr], label="datasheet")
    ps.ref_points(a, x, [r["eoff_datasheet_mJ"] for r in rr], label=None)
    a.set(xlabel=xlab, ylabel="Switching energy (mJ)", ylim=(0, None))
    a.set_title(f"{title}  (datasheet {figname})")
    a.legend()
fig.suptitle(f"Double-pulse simulation vs datasheet at 800 V  -  total switching energy within "
             f"{summary['etot_mean_abs_err_pct']:.0f} % (mean) on {len(val)} validation points",
             x=0.01, ha="left", fontsize=12, fontweight="bold", color=ps.INK)
fig.tight_layout()
ps.save(fig, "02_dpt_validation.png")

m, w = run(dev, cfg.with_(tj=25.0))
t = w["t"] * 1e9
fig, ax = plt.subplots(2, 2, figsize=(12, 6.4), sharex="col")
for col, (win, title) in enumerate(((m["t_off_window"], "Turn-off (end of pulse 1)"),
                                    (m["t_on_window"], "Turn-on (start of pulse 2)"))):
    t_a = win[0] * 1e9 - 30
    sel = (t >= t_a) & (t <= t_a + 200)
    tt = t[sel] - t_a
    a = ax[0, col]
    a.plot(tt, w["vds"][sel], color=ps.C1, label="Vds (V)")
    a.set_ylabel("Vds (V)")
    a.set_title(title)
    a = ax[1, col]
    a.plot(tt, w["id"][sel], color=ps.C2, label="Id (A)")
    a.set_ylabel("Id (A)")
    a.set_xlabel("time (ns)")
    lo, hi = (np.array(win) * 1e9 - t_a)
    for aa in ax[:, col]:
        aa.axvspan(lo, hi, color=ps.GRID, alpha=0.6, lw=0)
ax[0, 0].text(0.98, 0.05, f"Eoff = {m['eoff_mJ']:.2f} mJ\npeak {m['v_peak_V']:.0f} V",
              transform=ax[0, 0].transAxes, ha="right", va="bottom", color=ps.INK, fontsize=9)
ax[0, 1].text(0.98, 0.6, f"Eon = {m['eon_mJ']:.2f} mJ", transform=ax[0, 1].transAxes,
              ha="right", color=ps.INK, fontsize=9)
fig.suptitle("Simulated double-pulse waveforms, 800 V / 50 A, Rg = 2.5 Ω (shaded = energy window)",
             x=0.01, ha="left", fontsize=12, fontweight="bold", color=ps.INK)
fig.tight_layout()
ps.save(fig, "02_dpt_waveforms.png")
print("done")
