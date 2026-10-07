"""
Step 7 - worst-case analysis (WCA) and Monte Carlo.

Nominal results say the design works. A design review asks: does it still
work when every tolerance goes the wrong way, and how likely is trouble?

  1. Extreme-value overshoot: highest bus, +25 % loop inductance (layout
     spread), -1 % gate resistor, cold device, 10 % over-current -> peak Vds.
  2. Monte Carlo (5 000 builds) of the hottest junction at peak load with
     part-to-part and environmental spread.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sicinv.device import SiCMosfet                                  # noqa: E402
from sicinv.dpt import run                                           # noqa: E402
from sicinv.losses import EnergyTable, SiCPosition                   # noqa: E402
from sicinv.design import SPEC, CHOICE, per_device_loop, design_dpt  # noqa: E402
from sicinv import plotstyle as ps                                   # noqa: E402
from sicinv.plotstyle import plt                                     # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
dev = SiCMosfet.load(ps.DATA / "device_fit.json")
sw = json.loads((RES / "switching_design.json").read_text())
th = json.loads((RES / "losses.json").read_text())["thermal"]
etab = EnergyTable.load(RES / "sic_energy_table.json")
N = CHOICE["n_par"]
out = {}

# ------------------------------------------------- 1. worst-case overshoot --
cal = json.loads((ps.DATA / "dpt_calibration.json").read_text())
rgc = sw["gate_resistor_choice"]
RG_ON, RG_OFF, OPT = rgc["rg_on_ohm"], rgc["rg_off_ohm"], rgc["selected_option"]
I_PK = SPEC["i_peak_rms"] * np.sqrt(2) / N

corners = {
    "nominal (850 V, 25 C)": dict(),
    "+25 % loop inductance": dict(l_loop=per_device_loop(CHOICE["l_ext_nH"] * 1.25, N)),
    "+10 % over-current": dict(i_test=I_PK * 1.10),
    "cold device (-40 C)": dict(tj=-40.0),
    "all of the above": dict(l_loop=per_device_loop(CHOICE["l_ext_nH"] * 1.25, N),
                             i_test=I_PK * 1.10, tj=-40.0),
}
wc = {}
for name, kw in corners.items():
    m, _ = run(dev, design_dpt(cal, RG_ON * 0.99, RG_OFF * 0.99, OPT, **kw))
    wc[name] = {"v_peak_V": max(m["v_peak_V"], m["v_peak_hs_V"]),
                "dvdt_max_V_per_ns": max(m["dvdt_on_V_per_ns"], m["dvdt_off_V_per_ns"])}
    print(f"  {name:26s} Vpk {wc[name]['v_peak_V']:6.0f} V   dv/dt {wc[name]['dvdt_max_V_per_ns']:5.1f} V/ns",
          flush=True)
out["overshoot_corners"] = wc
out["overshoot_worst_V"] = max(v["v_peak_V"] for v in wc.values())
out["overshoot_margin_to_1200V_V"] = 1200 - out["overshoot_worst_V"]

# --------------------------------------------------------- 2. Monte Carlo --
rng = np.random.default_rng(2026)
K = 5000
share = max(max(s) for s in (sw["paralleling"]["threshold spread +-0.3 V"]["eon_share"],
                             sw["paralleling"]["threshold spread +-0.3 V"]["eoff_share"])) * N
pos = SiCPosition(dev, etab, n_par=N, t_dead=CHOICE["t_dead"])
samples = {
    # Rds(on): datasheet typ 21 / max 28.8 mOhm -> treat max as +3 sigma
    "rds_scale": rng.normal(1.0, (28.8 / 21 - 1) / 3, K).clip(0.85, None),
    # switching energy: model error (~7 %) + part spread
    "esw_scale": rng.normal(1.0, 0.10, K).clip(0.7, None),
    # thermal pad and cold-plate resistance: assembly spread +/- 20 % (3 sigma)
    "rth_scale": rng.normal(1.0, 0.20 / 3, K),
    # coolant inlet 65 C nominal, up to ~70 C
    "t_cool": rng.normal(SPEC["t_coolant"], 5.0 / 3, K),
    # dynamic current sharing: the worst device takes up to `share` x its fair part
    "share": rng.uniform(1.0, share, K),
}
rth_jc, rth_cs = th["rth_jc"], th["rth_case_coolant"]
i_rms = SPEC["i_peak_rms"]
phi = np.arccos(0.9)
def monte_carlo(fsw):
    """Hottest junction of each simulated build at peak load and switching frequency fsw."""
    t_grid = np.linspace(40, 220, 37)            # pre-compute losses vs temperature (fast lookup)
    p_cond, p_sw = [], []
    for t in t_grid:
        p = pos.losses(i_rms, SPEC["vdc_max"], fsw, t, 1.0, phi)
        p_cond.append((p["conduction"] + p["dead_time"]) / N)
        p_sw.append((p["turn_on"] + p["turn_off"] + p["recovery"]) / N)
    p_cond, p_sw = np.array(p_cond), np.array(p_sw)
    tj = np.empty(K)
    for k in range(K):
        rth = rth_jc + rth_cs * samples["rth_scale"][k]
        t = samples["t_cool"][k] + 40
        for _ in range(40):
            pc = np.interp(t, t_grid, p_cond) * samples["rds_scale"][k] * samples["share"][k] ** 2
            ps_ = np.interp(t, t_grid, p_sw) * samples["esw_scale"][k] * samples["share"][k]
            t_new = samples["t_cool"][k] + (pc + ps_) * rth
            if abs(t_new - t) < 0.05:
                break
            t = 0.5 * (t + t_new)
        tj[k] = t_new
    return tj


mc = {}
for fsw in (CHOICE["fsw"], CHOICE["fsw_peak"]):
    tj = monte_carlo(fsw)
    mc[f"{fsw / 1e3:.0f} kHz"] = {"tj": tj, "summary": {
        "n": K, "tj_mean_C": float(tj.mean()), "tj_p99_C": float(np.percentile(tj, 99)),
        "tj_max_C": float(tj.max()), "p_over_150C_pct": float(100 * np.mean(tj > 150)),
        "p_over_175C_pct": float(100 * np.mean(tj > 175))}}
    print(f"  Monte Carlo at {fsw / 1e3:.0f} kHz:", {k: round(v, 2) for k, v in mc[f"{fsw / 1e3:.0f} kHz"]["summary"].items()})
out["monte_carlo"] = {k: v["summary"] for k, v in mc.items()}
out["monte_carlo"]["share_max"] = share

fig, ax = plt.subplots(1, 2, figsize=(12, 4.0))
a = ax[0]
names = list(wc)
vals = [wc[n]["v_peak_V"] for n in names]
bars = a.barh(names, vals, color=ps.C1, edgecolor=ps.SURFACE, height=0.55)
a.bar_label(bars, fmt="%.0f V", fontsize=8.5, color=ps.INK, padding=3)
a.axvline(1200, color=ps.C8, lw=1.2)
a.text(1195, -0.6, "1200 V rating", ha="right", fontsize=8, color=ps.INK_2)
a.axvline(SPEC["vpeak_limit"], color=ps.INK_2, lw=1, ls=":")
a.set(xlim=(800, 1260), xlabel="peak Vds (V)", title="Worst-case overshoot corners")
a.invert_yaxis()
a.grid(axis="y", visible=False)
a = ax[1]
bins = np.linspace(90, 200, 56)
for (name, r), col in zip(mc.items(), (ps.C2, ps.C1)):
    a.hist(r["tj"], bins=bins, color=col, alpha=0.75, edgecolor=ps.SURFACE,
           label=f"{name}: P(Tj > 150 °C) = {r['summary']['p_over_150C_pct']:.1f} %")
a.axvline(SPEC["tj_limit"], color=ps.INK_2, lw=1, ls=":")
a.axvline(175, color=ps.C8, lw=1.2)
a.text(175, a.get_ylim()[1] * 0.92, " 175 °C max", fontsize=8, color=ps.INK_2)
a.set(xlabel="hottest junction at peak load (°C)", ylabel="builds",
      title=f"Monte Carlo, {K} builds at peak torque (300 A rms, 850 V)")
a.legend(loc="upper left")
fig.tight_layout()
ps.save(fig, "07_worst_case.png")
(RES / "worst_case.json").write_text(json.dumps(out, indent=2) + "\n")
