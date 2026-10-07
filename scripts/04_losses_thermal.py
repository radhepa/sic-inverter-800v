"""
Step 4 - inverter losses, thermal sizing and the SiC vs silicon-IGBT comparison.

  * junction-to-case thermal network fitted to the datasheet Zth curve
  * how many devices in parallel per switch (N) keep Tj <= 150 C
  * efficiency vs output power: SiC at 10/20 kHz vs IGBT at 10/20 kHz
  * where the losses go at the continuous rating

Both technologies use the same number of TO-247-4-class packages per switch
and the same cold plate, so the comparison is about the silicon, not the
mechanics. IGBT data: Infineon IKY75N120CH7 datasheet (1200 V / 75 A).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sicinv.device import SiCMosfet                                  # noqa: E402
from sicinv.losses import EnergyTable, SiCPosition, IGBTPosition     # noqa: E402
from sicinv.thermal import fit_foster, coldplate_path                # noqa: E402
from sicinv.design import SPEC, CHOICE                               # noqa: E402
from sicinv import plotstyle as ps                                   # noqa: E402
from sicinv.plotstyle import plt                                     # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
ds = json.loads((ps.DATA / "c3m0021120k_datasheet.json").read_text(encoding="utf-8"))
igbt_ds = json.loads((ps.DATA / "iky75n120ch7_datasheet.json").read_text(encoding="utf-8"))
dev = SiCMosfet.load(ps.DATA / "device_fit.json")
etab = EnergyTable.load(RES / "sic_energy_table.json")
N = CHOICE["n_par"]
FSW = CHOICE["fsw"]
out = {}

# ---------------------------------------------------------------- thermal --
zt, zz = np.array(ds["zth_jc_fig21_single_pulse"]["points"]).T
jc = fit_foster(zt, zz, n=4)
cs = coldplate_path()
RTH_SIC = jc.rth + cs.rth
out["thermal"] = {"foster_jc": {"r": jc.r, "tau": jc.tau}, "rth_jc": jc.rth,
                  "rth_case_coolant": cs.rth, "rth_total_per_device": RTH_SIC}
print(f"Rth j-c {jc.rth:.3f} K/W (datasheet 0.32), case-coolant {cs.rth:.2f}, total {RTH_SIC:.2f} K/W")

# ------------------------------------------------- how many in parallel? --
PHI = np.arccos(0.9)          # motor power factor 0.9
M = 1.0                       # modulation index near base speed
sizing = []
for n in range(2, 7):
    pos = SiCPosition(dev, etab, n_par=n, t_dead=CHOICE["t_dead"])
    row = {"n_par": n}
    for tag, i_rms in (("cont", SPEC["i_cont_rms"]), ("peak", SPEC["i_peak_rms"])):
        p = pos.steady_state(i_rms, SPEC["vdc_max"], FSW, RTH_SIC, SPEC["t_coolant"], M, PHI)
        row[f"tj_{tag}"] = p["tj"]
        row[f"loss_{tag}_W"] = 6 * p["total"] if not p["runaway"] else float("inf")
    sizing.append(row)
    print(f"  N={n}: Tj cont {row['tj_cont']:.0f} C, Tj peak {row['tj_peak']:.0f} C "
          f"(inf = thermal runaway), inverter loss at peak {row['loss_peak_W']:.0f} W")
n_min = min(r["n_par"] for r in sizing if r["tj_peak"] <= SPEC["tj_limit"])
out["sizing"] = {"rows": sizing, "n_min_meeting_tj_limit": n_min, "n_chosen": N}

fig, ax = plt.subplots(1, 2, figsize=(12, 4.0))
a = ax[0]
a.loglog(zt, zz, "o", mfc="none", mew=1.4, ms=7, color=ps.REF, label="datasheet (Fig. 21)")
tt = np.geomspace(1e-6, 1, 200)
a.loglog(tt, jc.zth(tt), color=ps.C1, label="4-term Foster fit")
a.set(title="Junction-to-case thermal impedance", xlabel="pulse length (s)", ylabel="Zth (K/W)")
a.legend()
a = ax[1]
ns = [r["n_par"] for r in sizing]
finite = [r for r in sizing if np.isfinite(r["tj_peak"])]
a.plot([r["n_par"] for r in finite], [r["tj_peak"] for r in finite], "o-", color=ps.C2, label="peak, 300 A rms")
a.plot([r["n_par"] for r in sizing if np.isfinite(r["tj_cont"])],
       [r["tj_cont"] for r in sizing if np.isfinite(r["tj_cont"])], "s-", color=ps.C1, label="continuous, 160 A rms")
for r in sizing:
    if not np.isfinite(r["tj_peak"]):
        a.annotate("thermal\nrunaway", (r["n_par"], 200), ha="center", fontsize=8, color=ps.INK_2)
a.axhline(SPEC["tj_limit"], color=ps.INK_2, lw=1, ls=":")
a.text(ns[-1], SPEC["tj_limit"], "design limit 150 °C", ha="right", va="bottom", fontsize=8, color=ps.INK_2)
a.axvline(N, color=ps.GRID, lw=8, zorder=0)
a.set(title=f"Devices per switch: {N} chosen (850 V, {FSW/1e3:.0f} kHz, 65 °C coolant)",
      xlabel="SiC MOSFETs in parallel per switch", ylabel="junction temperature (°C)")
a.set_xticks(ns)
a.legend()
fig.tight_layout()
ps.save(fig, "04a_thermal_sizing.png")

# ------------------------------------------------------ efficiency curves --
sic = SiCPosition(dev, etab, n_par=N, t_dead=CHOICE["t_dead"])
igbt = IGBTPosition(igbt_ds, n_par=N)
r_cs = cs.rth
V = SPEC["vdc_nom"]
v_ph = M * V / (2 * np.sqrt(2))                    # rms phase voltage
powers = np.linspace(5e3, 150e3, 30)
curves = {}
for name, f in (("SiC 20 kHz", 20e3), ("SiC 10 kHz", 10e3), ("IGBT 10 kHz", 10e3), ("IGBT 20 kHz", 20e3)):
    eff, tjs = [], []
    for pw in powers:
        i_rms = pw / (3 * v_ph * 0.9)
        if name.startswith("SiC"):
            p = sic.steady_state(i_rms, V, f, RTH_SIC, SPEC["t_coolant"], M, PHI)
        else:
            p = igbt.steady_state(i_rms, V, f, r_cs, SPEC["t_coolant"], M, PHI)
        loss = 6 * p["total"]
        eff.append(100 * pw / (pw + loss))
        tjs.append(p["tj"])
    curves[name] = {"p_kw": (powers / 1e3).tolist(), "eff_pct": eff, "tj": tjs}
out["efficiency_800V"] = curves

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
styles = {"SiC 20 kHz": (ps.C1, "-"), "SiC 10 kHz": (ps.C1, "--"),
          "IGBT 10 kHz": (ps.C2, "-"), "IGBT 20 kHz": (ps.C2, "--")}
for name, c in curves.items():
    col, ls = styles[name]
    ax[0].plot(c["p_kw"], c["eff_pct"], ls, color=col, label=name)
    ax[1].plot(c["p_kw"], c["tj"], ls, color=col, label=name)
ax[0].set(title="Inverter efficiency at 800 V", xlabel="output power (kW)", ylabel="efficiency (%)",
          ylim=(95, 100))
ax[1].axhline(SPEC["tj_limit"], color=ps.INK_2, lw=1, ls=":")
ax[1].set(title="Hottest junction", xlabel="output power (kW)", ylabel="°C")
for a in ax:
    a.legend()
fig.suptitle(f"Same packages, same cold plate: {N} SiC MOSFETs vs {N} Si IGBTs per switch",
             x=0.01, ha="left", fontsize=12, fontweight="bold", color=ps.INK)
fig.tight_layout()
ps.save(fig, "04b_efficiency.png")

# ---------------------------------------------------------- loss breakdown --
i_cont = SPEC["p_cont_kw"] * 1e3 / (3 * v_ph * 0.9)
p_s20 = sic.steady_state(i_cont, V, 20e3, RTH_SIC, SPEC["t_coolant"], M, PHI)
p_i10 = igbt.steady_state(i_cont, V, 10e3, r_cs, SPEC["t_coolant"], M, PHI)
bars = {
    "SiC, 20 kHz": {"conduction": p_s20["conduction"] + p_s20["dead_time"],
                    "switching": p_s20["turn_on"] + p_s20["turn_off"],
                    "diode recovery": p_s20["recovery"]},
    "Si IGBT, 10 kHz": {"conduction": p_i10["igbt_conduction"] + p_i10["diode_conduction"],
                        "switching": p_i10["igbt_turn_on"] + p_i10["igbt_turn_off"],
                        "diode recovery": p_i10["diode_recovery"]},
}
bars = {k: {kk: 6 * vv for kk, vv in v.items()} for k, v in bars.items()}
out["loss_breakdown_75kW_W"] = bars
fig, a = plt.subplots(figsize=(7.5, 3.6))
left = np.zeros(2)
names = list(bars)
for part, col in zip(("conduction", "switching", "diode recovery"), (ps.C1, ps.C2, ps.C3)):
    vals = np.array([bars[n][part] for n in names])
    a.barh(names, vals, left=left, color=col, edgecolor=ps.SURFACE, linewidth=2, label=part, height=0.55)
    left += vals
for k, n in enumerate(names):
    a.text(left[k] + 15, k, f"{left[k]:.0f} W", va="center", color=ps.INK, fontsize=9)
a.set(xlabel="inverter losses (W)", title=f"Where the losses go at the 75 kW continuous rating ({V:.0f} V)",
      xlim=(0, left.max() * 1.18))
a.legend(loc="lower right")
a.grid(axis="y", visible=False)
fig.tight_layout()
ps.save(fig, "04c_loss_breakdown.png")

summary = {
    "loss_75kW_sic20_W": sum(bars["SiC, 20 kHz"].values()),
    "loss_75kW_igbt10_W": sum(bars["Si IGBT, 10 kHz"].values()),
    "eff_75kW_sic20_pct": 100 * 75e3 / (75e3 + sum(bars["SiC, 20 kHz"].values())),
    "eff_75kW_igbt10_pct": 100 * 75e3 / (75e3 + sum(bars["Si IGBT, 10 kHz"].values())),
    "tj_75kW_sic20_C": p_s20["tj"], "tj_75kW_igbt10_C": p_i10["tj"],
    "peak_eff_sic20_pct": max(curves["SiC 20 kHz"]["eff_pct"]),
    "peak_eff_igbt10_pct": max(curves["IGBT 10 kHz"]["eff_pct"]),
}
out["summary"] = summary
(RES / "losses.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
for k, v in summary.items():
    print(f"  {k:24s} {v:8.2f}")
