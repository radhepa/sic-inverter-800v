"""
Step 5 - what SiC buys on the road: EPA city (UDDS) and highway (HWFET) cycles.

For every second of the speed trace: vehicle model -> motor torque/speed ->
inverter phase current -> losses at the current junction temperature ->
thermal model steps forward. Same packages, cold plate and vehicle for SiC
and for the silicon IGBT baseline.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sicinv.device import SiCMosfet                                  # noqa: E402
from sicinv.losses import EnergyTable, SiCPosition, IGBTPosition     # noqa: E402
from sicinv.thermal import coldplate_path                            # noqa: E402
from sicinv.vehicle import load_cycle, operating_points              # noqa: E402
from sicinv.design import SPEC, CHOICE                               # noqa: E402
from sicinv import plotstyle as ps                                   # noqa: E402
from sicinv.plotstyle import plt                                     # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
igbt_ds = json.loads((ps.DATA / "iky75n120ch7_datasheet.json").read_text(encoding="utf-8"))
dev = SiCMosfet.load(ps.DATA / "device_fit.json")
th = json.loads((RES / "losses.json").read_text())["thermal"]
N = CHOICE["n_par"]
V = SPEC["vdc_nom"]
sic = SiCPosition(dev, EnergyTable.load(RES / "sic_energy_table.json"), n_par=N, t_dead=CHOICE["t_dead"])
igbt = IGBTPosition(igbt_ds, n_par=N)
CS = coldplate_path()
RJC_SIC = th["rth_jc"]
RJC_I, RJC_D = igbt_ds["scalars"]["Rth_jc_igbt_K_per_W"], igbt_ds["scalars"]["Rth_jc_diode_K_per_W"]


def cold_plate_step(state, p, dt):
    """Case temperature rise above coolant through the pad + cold plate (Foster)."""
    a = np.exp(-dt / np.array(CS.tau))
    state[:] = state * a + p * np.array(CS.r) * (1 - a)
    return float(state.sum())


def simulate(op, tech, fsw):
    n = len(op["t"])
    loss = np.zeros(n)
    tj = np.full(n, SPEC["t_coolant"])
    state = np.zeros(len(CS.r))
    tj_prev = tj_i = tj_d = SPEC["t_coolant"]
    for k in range(n):
        dt = 1.0 if k == 0 else op["t"][k] - op["t"][k - 1]
        i_rms, m, phi = op["i_rms"][k], op["m"][k], op["phi"][k]
        if i_rms < 1.0:                          # standing still: PWM off
            dT = cold_plate_step(state, 0.0, dt)
            tj[k] = tj_prev = tj_i = tj_d = SPEC["t_coolant"] + dT
            continue
        if tech == "sic":
            p = sic.losses(i_rms, V, fsw, tj_prev, m, phi)
            p_dev = p["total"] / N
            dT = cold_plate_step(state, p_dev, dt)
            tj_prev = SPEC["t_coolant"] + dT + p_dev * RJC_SIC
            tj[k] = tj_prev
        else:
            p = igbt.losses(i_rms, V, fsw, tj_i, tj_d, m, phi)
            pi, pd = p["igbt"] / N, p["diode"] / N
            dT = cold_plate_step(state, pi + pd, dt)
            tj_i = SPEC["t_coolant"] + dT + pi * RJC_I
            tj_d = SPEC["t_coolant"] + dT + pd * RJC_D
            tj[k] = max(tj_i, tj_d)
        loss[k] = 6 * p["total"]
    return loss, tj


results = {}
traces = {}
for cyc, fname in (("UDDS (city)", "epa_udds.csv"), ("HWFET (highway)", "epa_hwfet.csv")):
    t, v = load_cycle(ps.DATA / fname)
    op = operating_points(t, v, V)
    dist_km = float(np.trapz(v, t)) / 1e3
    r = {"distance_km": dist_km}
    traces[cyc] = {"t": t, "v": v, "op": op}
    for tech, fsw, label in (("sic", 20e3, "SiC 20 kHz"), ("sic", 10e3, "SiC 10 kHz"),
                             ("igbt", 10e3, "IGBT 10 kHz")):
        loss, tj = simulate(op, tech, fsw)
        thru = float(np.trapz(np.abs(op["p_out"]), t))
        e_loss = float(np.trapz(loss, t))
        e_batt = float(np.trapz(op["p_out"] + loss, t))
        r[label] = {"inverter_loss_Wh": e_loss / 3600, "throughput_Wh": thru / 3600,
                    "cycle_efficiency_pct": 100 * thru / (thru + e_loss),
                    "battery_Wh_per_km": e_batt / 3600 / dist_km, "tj_max_C": float(tj.max())}
        traces[cyc][label] = (loss, tj)
    s, g = r["SiC 20 kHz"], r["IGBT 10 kHz"]
    r["loss_reduction_pct"] = 100 * (1 - s["inverter_loss_Wh"] / g["inverter_loss_Wh"])
    r["loss_reduction_same_fsw_pct"] = 100 * (1 - r["SiC 10 kHz"]["inverter_loss_Wh"] / g["inverter_loss_Wh"])
    r["range_gain_pct"] = 100 * (g["battery_Wh_per_km"] / s["battery_Wh_per_km"] - 1)
    results[cyc] = r
    print(f"{cyc}: {dist_km:.1f} km; inverter loss SiC {s['inverter_loss_Wh']:.0f} Wh vs IGBT "
          f"{g['inverter_loss_Wh']:.0f} Wh (-{r['loss_reduction_pct']:.0f} %), efficiency "
          f"{s['cycle_efficiency_pct']:.2f} % vs {g['cycle_efficiency_pct']:.2f} %, "
          f"range +{r['range_gain_pct']:.1f} %; at the same 10 kHz: -{r['loss_reduction_same_fsw_pct']:.0f} % loss")
(RES / "drive_cycle.json").write_text(json.dumps(results, indent=2) + "\n")

# ------------------------------------------------------------------ figures --
cyc = "UDDS (city)"
tr = traces[cyc]
fig, ax = plt.subplots(3, 1, figsize=(12, 7.6), sharex=True)
ax[0].plot(tr["t"], tr["v"] * 3.6, color=ps.INK_2, lw=1.2)
ax[0].set(ylabel="speed (km/h)", title="EPA city cycle (UDDS): speed, inverter losses, hottest junction")
for label, col in (("IGBT 10 kHz", ps.C2), ("SiC 20 kHz", ps.C1)):
    loss, tj = tr[label]
    ax[1].plot(tr["t"], loss / 1e3, color=col, lw=1.2, label=label)
    ax[2].plot(tr["t"], tj, color=col, lw=1.2, label=label)
ax[1].set(ylabel="inverter loss (kW)")
ax[2].set(ylabel="Tj (°C)", xlabel="time (s)")
ax[1].legend()
fig.tight_layout()
ps.save(fig, "05a_udds_trace.png")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.0))
cycles = list(results)
x = np.arange(len(cycles))
wbar = 0.27
bars_ = (("IGBT 10 kHz", ps.C2), ("SiC 10 kHz (same frequency)", ps.C3), ("SiC 20 kHz (design)", ps.C1))
for k, (label, col) in enumerate(bars_):
    key = label.split(" (")[0]
    eff = [results[c][key]["cycle_efficiency_pct"] for c in cycles]
    los = [results[c][key]["inverter_loss_Wh"] / results[c]["distance_km"] for c in cycles]
    b0 = ax[0].bar(x + (k - 1) * wbar, eff, wbar, color=col, edgecolor=ps.SURFACE, linewidth=2, label=label)
    b1 = ax[1].bar(x + (k - 1) * wbar, los, wbar, color=col, edgecolor=ps.SURFACE, linewidth=2, label=label)
    ax[0].bar_label(b0, fmt="%.1f %%", fontsize=8.5, color=ps.INK)
    ax[1].bar_label(b1, fmt="%.1f", fontsize=8.5, color=ps.INK)
lo = min(results[c]["IGBT 10 kHz"]["cycle_efficiency_pct"] for c in cycles)
ax[0].set(xticks=x, xticklabels=cycles, ylabel="inverter efficiency over the cycle (%)",
          ylim=(np.floor(lo) - 2, 100), title="Cycle efficiency")
ax[1].set(xticks=x, xticklabels=cycles, ylabel="inverter loss (Wh per km)", title="Energy lost in the inverter")
for a in ax:
    a.grid(axis="x", visible=False)
    a.legend()
fig.suptitle("SiC vs silicon IGBT over EPA drive cycles (800 V, same packages and cooling)",
             x=0.01, ha="left", fontsize=12, fontweight="bold", color=ps.INK)
fig.tight_layout()
ps.save(fig, "05b_cycle_summary.png")
