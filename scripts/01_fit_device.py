"""
Step 1 - fit the SiC MOSFET compact model to the datasheet's static curves.

Everything here is "static": I-V curves, capacitances, gate charge, body diode.
The dynamic calibration (switching energies) happens in 02_calibrate_dpt.py.

Output: data/device_fit.json  and  figures/01_device_fit.png
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sicinv.device import SiCMosfet                       # noqa: E402
from sicinv import plotstyle as ps                        # noqa: E402
from sicinv.plotstyle import plt                          # noqa: E402

ds = json.loads((ps.DATA / "c3m0021120k_datasheet.json").read_text(encoding="utf-8"))
sc = ds["scalars"]
dev = SiCMosfet()
report = {}


def pts(key, sub=None):
    block = ds[key] if sub is None else ds[key][sub]
    return np.array(block if isinstance(block, list) else block["points"], float).T


# --- 1. channel: transfer characteristic at Vds = 20 V ----------------------
# 25 C: threshold, transconductance AND the high-voltage threshold shift, which
# is pinned by the start of the Miller plateau in the gate-charge curve
# (Fig. 12: 20 A at 800 V needs Vgs = 6.3 V).
q_pts, vg_pts = pts("gate_charge_fig12")
vg, i_meas = pts("transfer_fig7_Vds20V", "25C")


def res_ch25(p):
    d = replace(dev, vt25=p[0], k25=p[1], dvt_hv=p[2])
    r = (d.id_channel(vg, 20.0 - i_meas * d.rd(25.0), 25.0) - i_meas) / i_meas
    v_plateau = d.vgs_for_current(20.0, 800.0, 25.0)
    return np.append(r, 2.0 * (v_plateau - vg_pts[1]))


fit = least_squares(res_ch25, [2.8, 7.0, 1.0], bounds=([0.5, 1.0, 0.0], [6.0, 30.0, 5.0]))
dev = replace(dev, vt25=fit.x[0], k25=fit.x[1], dvt_hv=fit.x[2])
report["transfer_rms_err_25C_pct"] = 100 * float(np.sqrt(np.mean(fit.fun[:-1] ** 2)))
report["plateau_err_V"] = float(fit.fun[-1] / 2.0)

vg, i_meas = pts("transfer_fig7_Vds20V", "175C")


def res_ch175(p):
    d = replace(dev, vt175=p[0], k175=p[1])
    return (d.id_channel(vg, 20.0 - i_meas * d.rd(175.0), 175.0) - i_meas) / i_meas


fit = least_squares(res_ch175, [2.4, 7.0], bounds=([0.5, 1.0], [6.0, 30.0]))
dev = replace(dev, vt175=fit.x[0], k175=fit.x[1])
report["transfer_rms_err_175C_pct"] = 100 * float(np.sqrt(np.mean(fit.fun ** 2)))

# --- 2. drift resistance vs temperature -------------------------------------
t_all, r_all = pts("rds_on_vs_T_fig4")
r_all = r_all * sc["Rds_on_25C_ohm"]
# fit over the inverter's operating range (coolant >= 25 C); below room
# temperature the channel term dominates and a simple model can't follow it
t_pts, r_target = t_all[t_all >= 25], r_all[t_all >= 25]


def res_rd(p):
    d = replace(dev, rd25=p[0], rd_exp=p[1])
    return np.array([d.rds_on(t) for t in t_pts]) / r_target - 1.0


fit = least_squares(res_rd, [0.01, 2.5], bounds=([1e-4, 0.0], [0.05, 5.0]))
dev = replace(dev, rd25=fit.x[0], rd_exp=fit.x[1])
report["rds_on_rms_err_pct"] = 100 * float(np.sqrt(np.mean(fit.fun ** 2)))

# --- 3. capacitances (Crss, Coss, Eoss) -------------------------------------
cap = ds["capacitance_fig17_18"]
v_rss, c_rss = np.array(cap["Crss"]).T
v_oss, c_oss = np.array(cap["Coss"]).T
v_eo, e_eo = pts("eoss_fig16")
CAP_KEYS = ["ca", "va", "ma", "cp", "vp", "wp", "cb", "vb", "mb", "cinf", "cq"]
SCALE = np.array([1e-12, 1, 1, 1e-12, 1, 1, 1e-12, 1, 1, 1e-12, 1e-12])


def with_caps(p):
    return replace(dev, **dict(zip(CAP_KEYS, p * SCALE)))


def res_caps(p):
    d = with_caps(p)
    r1 = np.log(d.crss(v_rss) / (c_rss * 1e-12))
    r2 = np.log(d.coss(v_oss) / (c_oss * 1e-12))
    r3 = 2.0 * (np.array([d.eoss(v) for v in v_eo]) / (e_eo * 1e-6) - 1.0)
    return np.concatenate([r1, r2, r3])


p0 = np.array([150, 4, 0.55, 400, 10, 1, 3000, 1, 0.55, 150, 300], float)
lo = np.array([1, 0.1, 0.1, 0, 5, 0.2, 10, 0.05, 0.1, 0, 0], float)
hi = np.array([5000, 200, 2.0, 3000, 20, 5, 20000, 100, 2.0, 400, 3000], float)
fit = least_squares(res_caps, p0, bounds=(lo, hi))
dev = with_caps(fit.x)
v_iss, c_iss = np.array(cap["Ciss"]).T
dev = replace(dev, cgs=float(np.mean(c_iss[v_iss >= 10] * 1e-12 - dev.crss(v_iss[v_iss >= 10]))))
report["cap_rms_log_err"] = float(np.sqrt(np.mean(fit.fun[: len(v_rss) + len(v_oss)] ** 2)))

# --- 4. gate charge: oxide part of Cgd (gate above drain) --------------------


def q_at(d, vg_level):
    q, vg = d.gate_charge(800.0, 20.0, 25.0)
    return float(np.interp(vg_level, vg, q))   # vg is monotonic


def res_qg(p):
    d = replace(dev, cacc=p[0] * 1e-9, vacc=p[1])
    return [q_at(d, 8.2) / 99e-9 - 1.0, q_at(d, 15.0) / 162e-9 - 1.0]


fit = least_squares(res_qg, [3.0, 2.0], bounds=([0.1, 0.1], [20.0, 20.0]))
dev = replace(dev, cacc=fit.x[0] * 1e-9, vacc=fit.x[1])
report["qg_total_nC"] = q_at(dev, 15.0) * 1e9

# --- 5. body diode -----------------------------------------------------------
i25, v25 = pts("body_diode_fig9_fig10", "25C")
i175, v175 = pts("body_diode_fig9_fig10", "175C")


DIO_KEYS = ["dio_is25", "dio_is175", "dio_n25", "dio_n175", "dio_rs"]


def with_dio(p):
    return replace(dev, **dict(zip(DIO_KEYS, [10 ** p[0], 10 ** p[1], p[2], p[3], p[4]])))


def res_dio(p):
    d = with_dio(p)
    return np.concatenate([d.vsd(i25, 25.0) - v25, d.vsd(i175, 175.0) - v175])


fit = least_squares(res_dio, [-6, -4, 9.7, 7.0, 0.004],
                    bounds=([-30, -30, 1, 1, 0], [0, 0, 40, 40, 0.05]))
dev = with_dio(fit.x)
report["diode_rms_err_V"] = float(np.sqrt(np.mean(fit.fun ** 2)))

# --- summary -------------------------------------------------------------------
summary = {
    "Rds_on_25C_mOhm": dev.rds_on(25) * 1e3,
    "Rds_on_175C_mOhm": dev.rds_on(175) * 1e3,
    "Ciss_1000V_pF": float(dev.ciss(1000.0)) * 1e12,
    "Coss_1000V_pF": float(dev.coss(1000.0)) * 1e12,
    "Crss_1000V_pF": float(dev.crss(1000.0)) * 1e12,
    "Eoss_800V_uJ": dev.eoss(800.0) * 1e6,
    "Eoss_1000V_uJ": dev.eoss(1000.0) * 1e6,
    "Qoss_800V_nC": dev.qoss(800.0) * 1e9,
    "Qg_0_to_15V_nC": report["qg_total_nC"],
    "Miller_plateau_20A_800V_V": dev.vgs_for_current(20.0, 800.0, 25.0),
    "Vsd_25A_25C_V": float(dev.vsd(25.0, 25.0)),
    "Vsd_25A_175C_V": float(dev.vsd(25.0, 175.0)),
}
dev.notes = {"fit_report": report, "summary": summary,
             "source": "static fit to C3M0021120K datasheet (scripts/01_fit_device.py)"}
dev.save(ps.DATA / "device_fit.json")

print("fit quality:")
for k, v in report.items():
    print(f"  {k:28s} {v:10.3f}")
print("model vs datasheet:")
for k, v in summary.items():
    print(f"  {k:28s} {v:10.2f}")

# --- figure ------------------------------------------------------------------
fig, ax = plt.subplots(2, 3, figsize=(13.5, 7.6))

a = ax[0, 0]
vg = np.linspace(2, 10.5, 200)
for tag, tj, col in (("25C", 25, ps.C1), ("175C", 175, ps.C2)):
    a.plot(vg, dev.id_channel(vg, 20.0, tj), color=col, label=f"model, {tj} °C")
    x, y = pts("transfer_fig7_Vds20V", tag)
    ps.ref_points(a, x, y, label="datasheet" if tj == 25 else None)
a.set(title="Transfer characteristic (Vds = 20 V)", xlabel="Vgs (V)", ylabel="Id (A)", ylim=(0, 210))
a.legend()

a = ax[0, 1]
tt = np.linspace(-50, 175, 60)
a.plot(tt, [dev.rds_on(t) * 1e3 for t in tt], color=ps.C1, label="model")
ps.ref_points(a, t_all, r_all * 1e3)
a.plot(tt, [dev.rd(t) * 1e3 for t in tt], color=ps.C3, lw=1.5, ls="--", label="drift-region part")
a.set(title="On-resistance vs temperature (Vgs = 15 V, 50 A)", xlabel="Tj (°C)", ylabel="Rds(on) (mΩ)")
a.legend()

a = ax[0, 2]
vv = np.geomspace(0.3, 1200, 400)
a.loglog(vv, dev.ciss(vv) * 1e12, color=ps.C1, label="Ciss")
a.loglog(vv, dev.coss(vv) * 1e12, color=ps.C2, label="Coss")
a.loglog(vv, dev.crss(vv) * 1e12, color=ps.C3, label="Crss")
for k in ("Ciss", "Coss", "Crss"):
    x, y = np.array(cap[k]).T
    ps.ref_points(a, x, y, label="datasheet" if k == "Ciss" else None)
a.set(title="Capacitances (Vgs = 0 V)", xlabel="Vds (V)", ylabel="pF")
a.legend(ncol=2)

a = ax[1, 0]
vv = np.linspace(0, 1200, 100)
a.plot(vv, [dev.eoss(v) * 1e6 for v in vv], color=ps.C1, label="model")
ps.ref_points(a, v_eo, e_eo)
a.set(title="Energy stored in Coss", xlabel="Vds (V)", ylabel="Eoss (µJ)")
a.legend()

a = ax[1, 1]
q, vgq = dev.gate_charge(800.0, 20.0, 25.0)
a.plot(q * 1e9, vgq, color=ps.C1, label="model")
ps.ref_points(a, q_pts, vg_pts)
a.set(title="Gate charge (800 V, 20 A)", xlabel="Qg (nC)", ylabel="Vgs (V)")
a.legend()

a = ax[1, 2]
ii = np.linspace(1, 200, 200)
for tag, tj, col in (("25C", 25, ps.C1), ("175C", 175, ps.C2)):
    a.plot(dev.vsd(ii, tj), ii, color=col, label=f"model, {tj} °C")
    x, y = pts("body_diode_fig9_fig10", tag)
    ps.ref_points(a, y, x, label="datasheet" if tj == 25 else None)
a.set(title="Body diode (Vgs = -4 V)", xlabel="Vsd (V)", ylabel="Isd (A)", xlim=(2, 8))
a.legend()

fig.suptitle("SiC MOSFET model fitted to the C3M0021120K datasheet", x=0.01, ha="left",
             fontsize=13, fontweight="bold", color=ps.INK)
fig.tight_layout()
print("wrote", ps.save(fig, "01_device_fit.png"))
