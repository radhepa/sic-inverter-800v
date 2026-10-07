"""
The design targets and the choices made, in one place. Every script reads
these values, so a change here flows through the whole analysis.
"""

SPEC = {
    # electrical
    "vdc_nom": 800.0,          # V, nominal DC link (800 V-class battery)
    "vdc_min": 550.0,          # V, empty battery
    "vdc_max": 850.0,          # V, full battery / regen
    "p_peak_kw": 150.0,        # peak electrical power (10 s)
    "p_cont_kw": 75.0,         # continuous power
    "i_peak_rms": 300.0,       # A rms phase current, peak torque (10 s)
    "i_cont_rms": 160.0,       # A rms phase current, continuous
    # environment
    "t_coolant": 65.0,         # C, 50/50 water-glycol inlet
    # limits I hold myself to
    "tj_limit": 150.0,         # C (device max is 175 C; keep 25 C margin)
    "vpeak_limit": 1100.0,     # V, max transient Vds at the worst corner (92 % of 1200 V)
    "dvdt_limit": 50.0,        # V/ns, turn-on dv/dt budget (motor insulation / EMI)
}

CHOICE = {
    "device": "C3M0021120K (1200 V, 21 mOhm, TO-247-4)",
    "n_par": 5,                # devices in parallel per switch -> 30 per inverter (step 4 sizing)
    "fsw": 20e3,               # Hz, normal driving (above the audible range)
    "fsw_peak": 10e3,          # Hz, at peak torque (> 200 A rms): limits Tj (step 7)
    "t_dead": 200e-9,          # s
    "vgs_on": 15.0,
    "vgs_off": -4.0,
    "l_ext_nH": 8.0,           # bulk DC-link capacitor -> switches (laminated busbar)
    "c_hf_uF": 1.0,            # local HF decoupling capacitors at each half bridge
    "l_hf_nH": 2.0,            # loop: HF capacitors -> switches (PCB planes)
    "esl_hf_nH": 1.0,
    "esr_hf_mOhm": 20.0,
    "r_damp_ohm": 25.0,        # per loop half (see DPTConfig.r_damp)
    "gate_driver": "UCC21750-Q1",
    # gate-driver output stage seen by the paralleled gates
    "drv_routh": 2.5,          # ohm, UCC21750-Q1 pull-up (datasheet)
    "drv_routl": 0.3,          # ohm, UCC21750-Q1 pull-down (datasheet)
    "booster_r": 0.5,          # ohm, option B: external totem-pole buffer
}

# Filled in by scripts/03_switching_design.py after the gate-resistor study.
RG_FILE = "design_gate_resistors.json"


def per_device_loop(l_ext_nH: float, n_par: int) -> float:
    """Commutation-loop inductance seen by ONE of n_par paralleled devices.

    n devices share the external loop, so the total current (n * i_dev) flows
    through it: L_ext * d(n i)/dt = (n * L_ext) * di_dev/dt.
    """
    return n_par * l_ext_nH * 1e-9


def driver_r_per_device(option: str) -> tuple[float, float]:
    """(source, sink) driver output resistance seen by each of the n_par gates.
    A = UCC21750-Q1 alone, B = with an external totem-pole booster."""
    n = CHOICE["n_par"]
    if option == "A":
        return n * CHOICE["drv_routh"], n * CHOICE["drv_routl"]
    return n * CHOICE["booster_r"], n * CHOICE["booster_r"]


def design_dpt(cal: dict, rg_on: float, rg_off: float, option: str, **kw):
    """Double-pulse configuration for ONE device as it sits in the inverter:
    worst-case bus voltage and peak current, shared impedances scaled by n_par."""
    from .dpt import DPTConfig
    n = CHOICE["n_par"]
    r_on, r_off = driver_r_per_device(option)
    cfg = DPTConfig(
        vdc=SPEC["vdc_max"], i_test=SPEC["i_peak_rms"] * 2 ** 0.5 / n, tj=25.0,
        rg_on=rg_on, rg_off=rg_off, r_drv=1e-3, r_drv_on=r_on, r_drv_off=r_off,
        vgs_on=CHOICE["vgs_on"], vgs_off=CHOICE["vgs_off"],
        l_loop=per_device_loop(CHOICE["l_ext_nH"], n), r_damp=n * CHOICE["r_damp_ohm"],
        c_hf=CHOICE["c_hf_uF"] * 1e-6 / n, l_hf=per_device_loop(CHOICE["l_hf_nH"], n),
        esl_hf=per_device_loop(CHOICE["esl_hf_nH"], n), esr_hf=n * CHOICE["esr_hf_mOhm"] * 1e-3,
        l_load=40e-6, c_load=200e-12 / n,           # ~200 pF of motor cable + winding
        edge=cal["edge_ns"] * 1e-9, t_gap=1.0e-6, tmax=0.5e-9,
        miller_clamp=True, clamp_vth=-2.0, clamp_r=1.0,
        hs_rg=rg_off + r_off)
    return cfg.with_(**kw)
