"""Run the whole analysis in order (about 30 minutes, mostly ngspice)."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["01_fit_device.py", "02_calibrate_dpt.py", "03_switching_design.py", "04_losses_thermal.py",
         "05_drive_cycle.py", "06_gate_driver.py", "07_worst_case.py"]

for step in STEPS:
    print(f"\n=== {step} ===", flush=True)
    subprocess.run([sys.executable, "-u", str(HERE / step)], check=True)
print("\nall steps done; figures/ and results/ are up to date")
