"""
Drive-cycle model: EPA speed trace -> wheel force -> motor torque/speed ->
inverter operating point (DC voltage, phase current, modulation index, power factor).

Vehicle: a mid-size 800 V EV sedan. Motor: a 150 kW permanent-magnet machine
with i_d = 0 control (torque proportional to current), which is accurate below
base speed; both EPA cycles stay below base speed. All parameters are
representative values, listed here so they can be challenged.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MPH = 0.44704  # m/s


@dataclass
class Vehicle:
    mass: float = 2100.0          # kg, including driver/test mass
    cd_a: float = 0.55            # drag coefficient x frontal area, m^2
    c_rr: float = 0.009           # rolling resistance
    rho: float = 1.2              # air density, kg/m^3
    r_wheel: float = 0.35         # m
    gear: float = 9.0             # motor : wheel
    eta_gear: float = 0.97
    rot_inertia: float = 1.05     # equivalent-mass factor for spinning parts
    regen_limit_kw: float = 100.0


@dataclass
class Motor:
    kt: float = 1.17              # N*m per A_rms (i_d = 0)
    pole_pairs: int = 4
    r_s: float = 0.015            # phase resistance, ohm
    l_q: float = 0.30e-3          # H
    p_iron_10krpm: float = 1200.0 # W of iron + friction at 10 000 rpm


def load_cycle(path: str | Path):
    rows = [r for r in csv.reader(open(path)) if r and not r[0].startswith("#")]
    t, v = np.array([[float(a), float(b)] for a, b in rows[1:]]).T
    return t, v * MPH


def operating_points(t, v, vdc: float, veh: Vehicle = Vehicle(), mot: Motor = Motor()):
    """Inverter operating point for every sample of a speed trace (1 s steps)."""
    a = np.gradient(v, t)
    f = (veh.rot_inertia * veh.mass * a + 0.5 * veh.rho * veh.cd_a * v ** 2
         + veh.c_rr * veh.mass * 9.81 * (v > 0.05))
    p_wheel = f * v
    w_motor = v / veh.r_wheel * veh.gear                          # rad/s
    # gearbox efficiency works against you in both directions
    p_shaft = np.where(p_wheel >= 0, p_wheel / veh.eta_gear, p_wheel * veh.eta_gear)
    p_shaft = np.maximum(p_shaft, -veh.regen_limit_kw * 1e3)      # friction brakes take the rest
    torque = np.divide(p_shaft, w_motor, out=np.zeros_like(p_shaft), where=w_motor > 1.0)
    i_rms = np.abs(torque) / mot.kt
    # phase voltage phasor: back-EMF + R*I + j*w_e*Lq*I  (I in phase with EMF)
    e_ph = mot.kt * w_motor / 3.0
    w_e = w_motor * mot.pole_pairs
    sgn = np.sign(torque)
    v_re = e_ph + sgn * mot.r_s * i_rms
    v_im = sgn * w_e * mot.l_q * i_rms
    v_ph = np.hypot(v_re, v_im)
    phi = np.arctan2(v_im, v_re)                                   # voltage leads current
    m = np.clip(v_ph * np.sqrt(2) / (vdc / 2), 0, 1.15)           # 1.15 = SVPWM limit
    p_iron = mot.p_iron_10krpm * (w_motor / (10000 * 2 * np.pi / 60)) ** 1.5
    p_elec = p_shaft + 3 * mot.r_s * i_rms ** 2 + p_iron * (w_motor > 1.0)
    return {"t": t, "v": v, "p_wheel": p_wheel, "torque": torque, "w_motor": w_motor,
            "i_rms": i_rms, "m": m, "phi": phi, "p_out": p_elec}
