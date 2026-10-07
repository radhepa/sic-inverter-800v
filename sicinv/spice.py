"""
Minimal ngspice driver: write a netlist, run ngspice in batch mode, read the
requested vectors back as numpy arrays.

ngspice is found via the NGSPICE environment variable, then the PATH
(`ngspice_con` / `ngspice`), then a few common install folders.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

_CANDIDATES = [
    Path.home() / "tools" / "Spice64" / "bin" / "ngspice_con.exe",
    Path("C:/Spice64/bin/ngspice_con.exe"),
    Path("C:/Program Files/Spice64/bin/ngspice_con.exe"),
]


class SpiceError(RuntimeError):
    pass


def find_ngspice() -> str:
    env = os.environ.get("NGSPICE")
    if env and Path(env).exists():
        return env
    for exe in ("ngspice_con", "ngspice"):
        hit = shutil.which(exe)
        if hit:
            return hit
    for p in _CANDIDATES:
        if p.exists():
            return str(p)
    raise SpiceError("ngspice not found: install it and/or set NGSPICE=<path to ngspice_con.exe>")


def run(netlist: str, vectors: list[str], timeout: float = 300.0,
        keep: str | Path | None = None, allow_fail: bool = False) -> dict[str, np.ndarray]:
    """Run a transient netlist (without .control block) and return vectors.

    `vectors` are ngspice expressions such as "v(sw)" or "i(vid)". The result
    dict also contains "time".
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        out = tmp / "out.txt"
        control = "\n".join([
            ".control",
            "set wr_singlescale",
            "set wr_vecnames",
            "option numdgt=9",
            "run",
            f"wrdata {out.as_posix()} " + " ".join(vectors),
            "quit",
            ".endc",
        ])
        body = netlist.rstrip()
        if body.lower().endswith(".end"):
            body = body[:-4].rstrip()
        text = f"{body}\n{control}\n.end\n"
        cir = tmp / "deck.cir"
        cir.write_text(text)
        if keep:
            Path(keep).write_text(text)
        try:
            proc = subprocess.run([find_ngspice(), "-b", str(cir)], capture_output=True,
                                  text=True, timeout=timeout, cwd=tmp)
        except subprocess.TimeoutExpired as exc:
            raise SpiceError(f"ngspice timed out after {timeout:.0f} s") from exc
        log = proc.stdout + proc.stderr
        failed = "aborted" in log or "Timestep too small" in log
        if not out.exists() or (failed and not allow_fail):
            bad = [ln for ln in log.splitlines() if "rror" in ln or "too small" in ln or "abort" in ln]
            raise SpiceError("ngspice run failed:\n" + "\n".join(bad[-6:] or [log[-1500:]]))
        data = np.loadtxt(out, skiprows=1)
        header = out.read_text().splitlines()[0].split()
    names = ["time"] + [v.lower() for v in vectors]
    if len(header) != len(names):
        raise SpiceError(f"unexpected wrdata header {header}")
    return {n: data[:, k] for k, n in enumerate(names)}
