# Design report: 800 V SiC traction inverter switch

This report walks through the design in the order it was done. Each section says what was
asked, how it was simulated, what came out, and what was decided. The numbers come from
`results/*.json`; the figures from `figures/`.

---

## 1. The job

A traction inverter turns the battery's DC into three-phase AC for the motor. Each phase has a
**half-bridge**: a high-side and a low-side switch that alternately connect the motor terminal
to the + or − rail tens of thousands of times a second (pulse-width modulation). This project
designs **one switch position**; the inverter has six identical ones.

| Requirement | Value | Why |
|---|---|---|
| DC link | 800 V nominal, 550–850 V | 800 V-class battery, empty to full |
| Peak phase current | 300 A<sub>rms</sub> (10 s) | peak motor torque |
| Continuous current | 160 A<sub>rms</sub> | continuous power (~75 kW) |
| Coolant | 65 °C water-glycol | typical EV cooling loop |
| Junction temperature | ≤ 150 °C | 25 °C margin to the device's 175 °C maximum |
| Peak V<sub>DS</sub> | ≤ 1100 V at the worst corner | margin to the 1200 V rating |
| Turn-on dv/dt | ≤ 50 V/ns | motor insulation and EMI |

**Why 800 V and SiC?** For the same power, doubling the voltage halves the current, so cables
and motor windings carry a quarter of the I²R loss and charging can run twice as fast. But
800 V needs 1200 V-rated switches. Silicon IGBTs at 1200 V switch slowly and have a fixed
~1 V "knee" voltage. Silicon-carbide MOSFETs switch about 10× faster and behave like a plain
resistor when on (no knee). This report quantifies how much that matters.

---

## 2. A SPICE model of the SiC MOSFET (`01_fit_device.py`)

Manufacturers publish curves, not equations, so the first job was to build a model that
reproduces the Wolfspeed **C3M0021120K** (1200 V, 21 mΩ, TO-247-4) datasheet.

**Equivalent circuit** (`sicinv/device.py`):

| Element | Physics | Fitted to |
|---|---|---|
| Channel current I<sub>D</sub>(V<sub>GS</sub>, V<sub>DS</sub>) | square-law MOSFET, smooth turn-on, conducts both ways | transfer curve (Fig. 7) |
| Drift resistance R<sub>d</sub>(T) ∝ T<sup>2.34</sup> | electron mobility in SiC falls with temperature | R<sub>DS(on)</sub> vs T (Fig. 4) |
| C<sub>gs</sub>, C<sub>gd</sub>(V), C<sub>ds</sub>(V) | oxide and depletion capacitances | C<sub>iss</sub>/C<sub>oss</sub>/C<sub>rss</sub> (Fig. 17–18), E<sub>oss</sub> (Fig. 16), gate charge (Fig. 12) |
| Body diode | PiN diode, ~3 V knee, carrier lifetime | V<sub>SD</sub> curves (Fig. 9–10), Q<sub>rr</sub> |
| R<sub>g,int</sub> = 3.3 Ω, package inductances | | datasheet table, typical TO-247-4 values |

**Result** ([figure](../figures/01_device_fit.png)): R<sub>DS(on)</sub> 20.2 / 35.1 mΩ at
25 / 175 °C (datasheet curve 21 / 36), total gate charge 162 nC (162), E<sub>oss</sub> at 800 V
70.6 µJ (70), body-diode V<sub>SD</sub> within 0.02 V.

**One honest correction.** The first model switched about 30 % too fast. Checking the internals
showed why: the datasheet's gate-charge curve puts the Miller plateau at 6.3 V for 20 A at
800 V, while the 20 V transfer curve predicts 5.2 V. I added a threshold shift that grows with
V<sub>DS</sub>, fitted to that one point. With it, the fitted thresholds came out at
**2.53 V and 1.98 V**, matching the datasheet's specified V<sub>th</sub> of 2.5 V and 2.0 V,
which were never used in the fit. That's good independent evidence the model is physically
right.

---

## 3. Double-pulse test (`02_calibrate_dpt.py`)

The **double-pulse test** is how the industry measures switching loss. An inductor is charged to
the test current by a first pulse. The device turns off at that current (E<sub>off</sub>), the
current freewheels through the other device's diode, and a second pulse turns the device back
on into the full current (E<sub>on</sub>). Energy is the integral of V<sub>DS</sub>·I<sub>D</sub>
over each transition (`sicinv/dpt.py`).

The datasheet doesn't describe its test fixture, so five unknowns were fitted at **one**
operating point (800 V, 50 A, R<sub>g</sub> = 2.5 Ω, 25 °C and 175 °C): gate-driver resistance,
driver edge time, power-loop inductance, load-inductor capacitance, and the body diode's carrier
lifetime. The lifetime is anchored to the datasheet's Q<sub>rr</sub> = 928 nC; the model gives
1043 nC.

**Validation on 13 points not used in calibration** ([figure](../figures/02_dpt_validation.png)):

| Sweep | Mean error (E<sub>on</sub>+E<sub>off</sub>) | Worst |
|---|---|---|
| Current, 10–80 A (50 A is the calibration point) | 3.6 % | 5.7 % |
| Temperature, 75–125 °C | 1.8 % | 2.2 % |
| Gate resistor, 5–20 Ω | 20.6 % | 24.5 % (at 20 Ω) |

The model under-predicts how much a large gate resistor slows the switching. The final design
uses 4.7 and 6.8 Ω, where the error is around 10–15 %. The Monte Carlo in section 8 adds a
10 % energy uncertainty to cover it.

---

## 4. Switching design (`03_switching_design.py`)

Simulated per device, at the worst corner for overshoot: 850 V bus, peak current
(300 A<sub>rms</sub> × √2 / 5 = 85 A per device). Five devices share the loop inductance and
the gate driver, so each device "sees" 5× the shared inductance and driver resistance
(`sicinv/design.py`, `per_device_loop`). Study E then simulates all five explicitly to check
that shortcut.

### 4.1 Gate resistors

R<sub>g</sub> sets the speed. Faster switching means less loss but more voltage overshoot
(V = L·di/dt) and more dv/dt ([figure](../figures/03a_gate_resistor.png)).

- **Turn-off resistor:** the smallest standard value with peak V<sub>DS</sub> ≤ 1100 V is **4.7 Ω** (1085 V).
- **Turn-on resistor:** the smallest value with turn-on dv/dt ≤ 50 V/ns is **6.8 Ω**.
- **Booster or not?** A gate-driver booster (lower driver resistance) was simulated as option B.
  It needs 22 Ω to meet the dv/dt limit and ends up with *more* loss (4.7 vs 3.9 mJ per device).
  When dv/dt is the binding constraint, a stronger driver buys nothing. **Decision: the UCC21750-Q1 alone.**
- Turn-off dv/dt at peak current is about 90 V/ns whatever the gate does: the load current
  charges the output capacitances (dv/dt ≈ I / 2C<sub>oss</sub>). The motor insulation must be
  rated for it. That's a system-level requirement this study passes on.

### 4.2 Layout inductance

With only an 8 nH DC-link busbar, staying under 1100 V needed a 22 Ω turn-off resistor and
about 3× the turn-off loss. Adding **1 µF of local high-frequency capacitors** right at the
switches leaves only the small loop from those capacitors to the devices as the fast loop. The
sweep ([figure](../figures/03b_loop_inductance.png)) gives the **PCB layout requirement:
≤ 2.5 nH** (design target 2 nH). This is the main reason production 800 V inverters use
low-inductance power modules.

### 4.3 Crosstalk, negative gate bias, Miller clamp

When the low-side device turns on, the high-side device's drain voltage swings by 850 V in
nanoseconds. That dv/dt pushes current through the high-side device's gate-drain capacitance
into its gate resistor and lifts its gate voltage. If the gate passes threshold, both devices
conduct (shoot-through).

| High-side off state | Peak gate voltage (175 °C, V<sub>th</sub> = 2.0 V) |
|---|---|
| 0 V, no clamp | **+2.9 V: false turn-on** |
| −4 V, no clamp | −1.7 V |
| −4 V + active Miller clamp | **−3.4 V** |

**Decision: +15 / −4 V gate drive with a Miller clamp.** With five paralleled gates, the
driver's single clamp pin can't reach every gate through its own resistor, so each gate gets a
Schottky diode to the shared clamp.

### 4.4 Paralleling

With a ±0.3 V threshold spread across the five devices, the lowest-threshold device turns on
first and off last. It takes 22 % of the switching energy instead of 20 %, and 6 % more peak
current ([figure](../figures/03e_paralleling.png)). Mitigations: an individual gate resistor
per device, Kelvin-source resistors (0.5 Ω), a symmetrical layout, and the 1.1× sharing factor
carried into the Monte Carlo.

### 4.5 Short circuit and DESAT

In a shoot-through the current is limited only by the device itself: about 470 A per device in
the (isothermal, so pessimistic) model. SiC dies are small and can survive only about 2 µs.
The UCC21750-Q1's **desaturation (DESAT)** input notices V<sub>DS</sub> staying high while
the gate is on, waits out a blanking time, then performs a slow, controlled turn-off with
400 mA, shared by five gates. A fast hard turn-off at 470 A per device would create a
destructive L·di/dt spike.

The delay chain from the datasheet is: 200 ns blanking, then C<sub>blk</sub>·V<sub>th</sub>/I<sub>chg</sub>,
then a 140 ns filter, then 200 ns to the output.

| C<sub>blk</sub> | Trip (typ / worst) | Current gone (typ / worst) |
|---|---|---|
| 22 pF | 0.94 / 1.23 µs | 1.57 / 2.24 µs |
| **10 pF** | **0.72 / 0.96 µs** | **1.35 / 1.97 µs** |

**Decision: 10 pF.** The DESAT diodes are two 1200 V parts in series so they can block the
850 V bus. The trip level of 6.9–8.8 V corresponds to 211–265 A per device at 150 °C, while
normal peak current gives only 2.7 V, so false trips can't happen.

---

## 5. Losses and thermal design (`04_losses_thermal.py`, `sicinv/losses.py`)

The loss model averages each device's losses over one electrical cycle (720 points). For a SiC
MOSFET with **synchronous rectification** (the channel also conducts backwards, so the body
diode carries current only during the 200 ns dead time):

- conduction: duty × V<sub>DS,on</sub>(i, T<sub>j</sub>) × |i|, where V<sub>DS,on</sub> comes from the device model
- switching: f<sub>sw</sub> × (E<sub>on</sub> + E<sub>off</sub>) at the instantaneous current, from a 54-point energy table simulated at the design gate resistors and layout (`results/sic_energy_table.json`)
- diode: dead-time conduction plus recovery energy

Because losses raise T<sub>j</sub>, and R<sub>DS(on)</sub> rises with T<sub>j</sub>, the model
iterates to a self-consistent temperature. If losses grow faster than the cooling can remove
them, it reports **thermal runaway**.

**Thermal path per device:** junction to case, a 4-term Foster network fitted to the
datasheet's Z<sub>th</sub> curve (0.32 K/W), then case to coolant through a 0.22 K/W insulating
pad and a 0.30 K/W cold plate.

**Device count** ([figure](../figures/04a_thermal_sizing.png)), at the 850 V bus, 20 kHz and peak current:

| Per switch | Peak T<sub>j</sub> |
|---|---|
| 2 or 3 | thermal runaway |
| 4 | 182 °C |
| **5** | **134 °C** |
| 6 | 113 °C |

**SiC vs silicon IGBT.** The baseline is Infineon's IKY75N120CH7 (1200 V, 75 A, same package
class), with the same number of packages and the same cold plate. Its energies are scaled
linearly with voltage, which flatters the IGBT. At 75 kW: **SiC (20 kHz) 402 W vs IGBT (10 kHz) 720 W**
([figure](../figures/04c_loss_breakdown.png)). The IGBT loses on conduction because of its
~0.6–0.9 V knee, and on switching and diode recovery because of its tail current and
bipolar diode.

---

## 6. Drive cycles (`05_drive_cycle.py`)

The EPA city (UDDS) and highway (HWFET) speed traces drive a 2100 kg sedan model. That gives
the motor torque and speed, which give the phase current, inverter losses, and a thermal
step every second.

| | IGBT 10 kHz | SiC 10 kHz | SiC 20 kHz |
|---|---|---|---|
| UDDS inverter loss | 83 Wh | 35 Wh (−58 %) | 59 Wh (−29 %) |
| HWFET inverter loss | 27 Wh | 15 Wh (−45 %) | 28 Wh (+3 %) |

At the same switching frequency, SiC removes about half the inverter's drive-cycle loss. Its
gain is largest in city driving, which happens mostly at light load, where the IGBT's fixed
knee voltage hurts most. Running SiC at 20 kHz keeps the switching noise above hearing and
lowers current ripple, but it spends much of that gain. That's a real design trade-off.

---

## 7. Gate-driver design (`06_gate_driver.py`)

| Quantity | Value | Check |
|---|---|---|
| Gate charge per device (−4 → 15 V, 800 V, 85 A) | 159 nC | from the device model |
| Gate-drive power per switch at 20 kHz | 0.30 W | 2 W isolated supply: 5.3× margin |
| Peak source / sink current | 4.2 A / 10 A | driver rated 10 A: turn-off is driver-limited |
| Peak dv/dt vs isolation barrier CMTI | 89 V/ns vs ≥ 150 V/ns | 1.7× margin |
| DESAT trip vs normal V<sub>DS</sub> | 6.9 V minimum vs 2.7 V | 4.2 V margin, no false trips |
| R<sub>DS(on)</sub> at V<sub>GS</sub> = 12 V (UVLO) vs 15 V, 150 °C | 35.1 vs 32.0 mΩ | UVLO stops linear-mode operation |
| Temperature sense | AIN 200 µA into 10 k NTC + 3.3 k | 2.66 V at 25 °C, 0.73 V at 150 °C, inside AIN's 0.6–4.5 V range |
| Isolation | reinforced, 5.7 kV<sub>rms</sub>, 1.5 kV<sub>rms</sub> working | 850 V bus |

Bias supply: SN6505B-Q1 push-pull transformer driver, then a 1:4.5 transformer, a rectifier,
an LDO set to 19 V, and a 3.9 V Zener that places the source (COM) 3.9 V above VEE. That
yields **+15.1 V / −3.9 V**, inside the datasheet's −4/+15 V recommendation.

---

## 8. Worst case and Monte Carlo (`07_worst_case.py`)

**Overshoot corners** at the 850 V bus with R<sub>g</sub> −1 %:

| Corner | Peak V<sub>DS</sub> |
|---|---|
| nominal | 1085 V |
| +25 % loop inductance | 1086 V |
| +10 % over-current | 1097 V |
| −40 °C | 1087 V |
| all at once | **1101 V** (99 V below the rating) |

**Monte Carlo, 5,000 simulated builds at peak torque.** Varied parameters: R<sub>DS(on)</sub>
(the datasheet maximum treated as +3σ), switching energy ±10 %, thermal interface ±20 %,
coolant up to ~70 °C, and current sharing up to 1.1×.

| Switching frequency | Mean T<sub>j</sub> | 99th percentile | P(T<sub>j</sub> > 150 °C) | P(T<sub>j</sub> > 175 °C) |
|---|---|---|---|---|
| 20 kHz | 143 °C | 173 °C | **24 %** | 0.6 % |
| 10 kHz | 127 °C | 152 °C | **1.6 %** | 0 % |

The nominal design met 150 °C, but a quarter of real builds wouldn't. **Decision: drop to
10 kHz at peak torque** (variable switching frequency is standard practice). The remaining
1.6 % is covered by the NTC reading: the controller derates current when it measures high
temperature.

---

## 9. Design FMEA (abbreviated)

Severity, occurrence and detection are each scored 1–10. RPN = S × O × D.

| Failure mode | Effect | Cause | Prevention / detection | S | O | D | RPN |
|---|---|---|---|---|---|---|---|
| Shoot-through | device destroyed | false turn-on by dv/dt; control fault | −4 V off bias, Miller clamp, dead time; DESAT | 9 | 2 | 2 | 36 |
| Overshoot > 1200 V | avalanche, device failure | layout inductance above budget | ≤ 2.5 nH layout requirement, local HF caps, R<sub>g,off</sub> 4.7 Ω, WCA | 9 | 2 | 4 | 72 |
| T<sub>j</sub> > 175 °C | wear-out, failure | poor TIM, current imbalance, high coolant | N = 5, 10 kHz at peak, NTC + derating, Monte Carlo | 8 | 3 | 3 | 72 |
| Current imbalance | one device overheats | V<sub>th</sub> spread, layout asymmetry | individual R<sub>g</sub>, Kelvin resistors, symmetric layout | 6 | 4 | 5 | 120 |
| Gate over-voltage | gate oxide damage | ringing, supply fault | BAT54S clamps gate to VDD/VEE, regulated bias | 8 | 2 | 3 | 48 |
| Bias supply undervoltage | linear-mode heating | supply fault | driver UVLO (12 V), RDY to controller | 7 | 2 | 2 | 28 |
| Isolation breakdown | HV on LV side (safety) | creepage too small, contamination | reinforced driver, ≥ 8 mm creepage, PCB slot, conformal coat | 10 | 1 | 4 | 40 |
| DESAT false trip | unwanted shutdown | noise at turn-on | blanking + 200 ns leading-edge blank, 4.2 V margin | 4 | 3 | 3 | 36 |

The highest RPN is **current imbalance**. That's why the next step is the PCB layout with a
symmetric gate and power path, followed by a measurement.

---

## 10. Schematic (`hardware/gate-driver/`)

One switch, drawn in KiCad by `hardware/gen_gate_driver.py` from the results. The values can't
drift from the analysis, and the generator refuses to write a sheet where two different nets
touch. It covers the low-voltage interface (PWM, enable, fault, ready, temperature PWM), the
isolated bias supply, the UCC21750-Q1 with DESAT and the Miller clamp, the NTC, and five
C3M0021120K with their gate networks. KiCad ERC reports 0 errors. The remaining warnings are
about the library configuration on the build machine.

---

## 11. What I would do next

1. PCB layout in KiCad against the 2.5 nH loop budget, with symmetric gate traces and ≥ 8 mm creepage.
2. Double-pulse test on real hardware to replace the datasheet calibration with measurements.
3. A cross-check in LTspice with the manufacturer's SPICE model.
4. Electro-thermal short-circuit simulation, so the current drops as the die heats.
