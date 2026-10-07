"""
Step 6 - gate-driver design calculations for one switch (N paralleled SiC
MOSFETs driven by one UCC21750-Q1 isolated gate driver).

Every number is computed from the device model, the chosen gate resistors
(step 3) and the driver datasheet, and written to results/gate_driver.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sicinv.device import SiCMosfet                     # noqa: E402
from sicinv.design import SPEC, CHOICE                  # noqa: E402
from sicinv import shortcircuit as sc                   # noqa: E402
from sicinv import plotstyle as ps                      # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
dev = SiCMosfet.load(ps.DATA / "device_fit.json")
sw = json.loads((RES / "switching_design.json").read_text())
N, FSW = CHOICE["n_par"], CHOICE["fsw"]
VON, VOFF = CHOICE["vgs_on"], CHOICE["vgs_off"]
DV = VON - VOFF
rgc = sw["gate_resistor_choice"]
RG_ON, RG_OFF, OPT = rgc["rg_on_ohm"], rgc["rg_off_ohm"], rgc["selected_option"]
R_SRC = CHOICE["booster_r"] if OPT == "B" else CHOICE["drv_routh"]
R_SNK = CHOICE["booster_r"] if OPT == "B" else CHOICE["drv_routl"]
I_PK_DEV = SPEC["i_peak_rms"] * np.sqrt(2) / N
calc = {}

# 1. gate charge ------------------------------------------------------------
q, vg = dev.gate_charge(800.0, I_PK_DEV, 25.0, VOFF, VON)
qg = float(q[-1])
calc["Qg_per_device_nC"] = qg * 1e9
calc["Qg_per_switch_nC"] = N * qg * 1e9

# 2. drive power ------------------------------------------------------------
p_gate = N * qg * DV * FSW
p_q = 4e-3 * DV                                   # VDD quiescent current ~4 mA (datasheet)
calc["gate_power_per_switch_W"] = p_gate
calc["driver_quiescent_W"] = p_q
calc["isolated_supply_needed_W"] = p_gate + p_q
calc["isolated_supply_chosen_W"] = 2.0
calc["supply_margin_x"] = 2.0 / (p_gate + p_q)

# 3. peak gate currents (driver must stay under its 10 A rating) ------------
r_int = dev.rg_int
calc["peak_source_A"] = DV / (R_SRC + (RG_ON + r_int) / N)
calc["peak_sink_A"] = DV / (R_SNK + (RG_OFF + r_int) / N)
calc["driver_peak_rating_A"] = 10.0

# 4. gate-resistor dissipation (average + worst pulse) ----------------------
e_cycle = qg * DV                                  # energy per device per charge or discharge
calc["P_Rg_on_avg_W"] = 0.5 * e_cycle * FSW * RG_ON / (RG_ON + r_int + N * R_SRC)
calc["P_Rg_off_avg_W"] = 0.5 * e_cycle * FSW * RG_OFF / (RG_OFF + r_int + N * R_SNK)
calc["Rg_peak_pulse_W"] = max((calc["peak_source_A"] / N) ** 2 * RG_ON,
                              (calc["peak_sink_A"] / N) ** 2 * RG_OFF)

# 5. decoupling: supply droop during one turn-on < 0.5 V ---------------------
calc["C_bulk_min_uF"] = N * qg / 0.5 * 1e6
calc["C_bulk_chosen"] = "10 uF X7R 50 V + 100 nF per rail at the pins (datasheet asks > 10 uF)"

# 6. common-mode transient immunity -----------------------------------------
dv_max = max(max(r["dvdt_on_V_per_ns"], r["dvdt_off_V_per_ns"])
             for r in sw["gate_resistor_sweep"][OPT] if r["rg"] in (RG_ON, RG_OFF))
calc["max_dvdt_V_per_ns"] = dv_max
calc["CMTI_min_V_per_ns"] = 150.0
calc["CMTI_margin_x"] = 150.0 / dv_max

# 7. DESAT short-circuit detection ------------------------------------------
R_DS, VF = 1e3, (0.6, 0.8, 1.0)                   # series resistor; DESAT diode Vf min/typ/max
v_trip = {"min": 8.5 - VF[2] - 570e-6 * R_DS,
          "typ": 9.15 - VF[1] - 500e-6 * R_DS,
          "max": 9.8 - VF[0] - 430e-6 * R_DS}


def current_at_vds(v, tj):
    lo, hi = 0.0, 2000.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if dev.vds_on(mid, tj) < v else (lo, mid)
    return 0.5 * (lo + hi)


calc["desat_trip_Vds_V"] = v_trip
calc["vds_at_peak_current_150C_V"] = dev.vds_on(I_PK_DEV, 150.0)
calc["desat_trip_current_per_device_150C_A"] = {k: current_at_vds(v, 150.0) for k, v in v_trip.items()}
calc["desat_false_trip_margin_V"] = v_trip["min"] - calc["vds_at_peak_current_150C_V"]
c_blk = sw["short_circuit"]["c_blk_pF"] * 1e-12
calc["desat_blanking_cap_pF"] = c_blk * 1e12
calc["desat_delay_us"] = {"typ": sc.desat_delay(c_blk) * 1e6,
                          "worst": sc.desat_delay(c_blk, sc.DESAT_WORST) * 1e6}
calc["short_circuit_results"] = sw["short_circuit"]["runs"]

# 8. undervoltage lockout: what Vgs = 12 V would mean -------------------------
calc["Rds_on_150C_at_15V_mOhm"] = dev.rds_on(150.0, 50.0, 15.0) * 1e3
calc["Rds_on_150C_at_12V_mOhm"] = dev.rds_on(150.0, 50.0, 12.0) * 1e3
calc["UVLO_on_off_V"] = (12.0, 10.7)

# 9. temperature sense through the isolated AIN channel ----------------------
I_AIN, R_SER, R25, B = 200e-6, 3.3e3, 10e3, 3435.0


def r_ntc(t_c):
    return R25 * np.exp(B * (1 / (t_c + 273.15) - 1 / 298.15))


calc["AIN_V_at"] = {f"{t} C": I_AIN * (r_ntc(t) + R_SER) for t in (25, 65, 100, 125, 150)}
calc["AIN_note"] = "10 k NTC (B 3435) + 3.3 k series on the cold plate; AIN range 0.6-4.5 V"

# 10. isolation ----------------------------------------------------------------
calc["isolation"] = {"type": "reinforced", "viso_Vrms": 5700, "viowm_Vrms": 1500,
                     "viowm_Vdc_equiv": 1500 * np.sqrt(2), "bus_max_Vdc": SPEC["vdc_max"],
                     "package_creepage_mm": 8.0}

(RES / "gate_driver.json").write_text(json.dumps(calc, indent=2, default=float) + "\n")
w = max(len(k) for k in calc)
for k, v in calc.items():
    if isinstance(v, float):
        print(f"  {k:{w}s} {v:10.3f}")
    else:
        print(f"  {k:{w}s} {v}")
