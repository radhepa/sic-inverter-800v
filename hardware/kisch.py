"""
Tiny KiCad schematic writer: just enough to build a clean, ERC-checked
.kicad_sch from Python.

* symbols come from KiCad's own libraries (or a project library), copied into
  the schematic's lib_symbols block as KiCad requires
* every pin is wired with a short stub and a net label, so connectivity is
  explicit and easy to check
"""
from __future__ import annotations

import math
import os
import re
import uuid as _uuid
from pathlib import Path

def _kicad_symbols() -> Path:
    """KiCad's symbol library folder (set KICAD10_SYMBOL_DIR to override)."""
    env = os.environ.get("KICAD10_SYMBOL_DIR") or os.environ.get("KICAD_SYMBOL_DIR")
    candidates = [Path(env)] if env else []
    candidates += [Path.home() / "AppData/Local/Programs/KiCad/10.0/share/kicad/symbols",
                   Path("C:/Program Files/KiCad/10.0/share/kicad/symbols"),
                   Path("/usr/share/kicad/symbols"),
                   Path("/Applications/KiCad/KiCad.app/Contents/SharedSupport/symbols")]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("KiCad symbol libraries not found: set KICAD10_SYMBOL_DIR")


KICAD_SYMBOLS = _kicad_symbols()


# ----------------------------------------------------------- s-expressions --
def parse(text: str):
    tokens = re.findall(r'"(?:\\.|[^"\\])*"|\(|\)|[^\s()"]+', text)
    stack, cur = [], []
    for t in tokens:
        if t == "(":
            stack.append(cur)
            cur = []
        elif t == ")":
            done = cur
            cur = stack.pop()
            cur.append(done)
        else:
            cur.append(t)
    return cur[0]


def dump(x, indent=0) -> str:
    if not isinstance(x, list):
        return x
    pad = "\t" * indent
    simple = all(not isinstance(e, list) for e in x)
    if simple:
        return pad + "(" + " ".join(x) + ")"
    head = [e for e in x if not isinstance(e, list)]
    out = pad + "(" + " ".join(head)
    for e in x:
        if isinstance(e, list):
            out += "\n" + dump(e, indent + 1)
    return out + "\n" + pad + ")"


def q(s: str) -> str:
    return '"' + s.replace('"', '\\"') + '"'


def unq(s: str) -> str:
    return s[1:-1] if s.startswith('"') else s


def find(x, key):
    return [e for e in x if isinstance(e, list) and e and e[0] == key]


def snap(v: float, grid: float = 2.54) -> float:
    return round(round(v / grid) * grid, 2)


def uid() -> str:
    return str(_uuid.uuid4())


# ------------------------------------------------------------ symbol libs --
_lib_cache: dict[str, list] = {}


def load_lib(path: Path) -> list:
    key = str(path)
    if key not in _lib_cache:
        _lib_cache[key] = parse(path.read_text(encoding="utf-8"))
    return _lib_cache[key]


def get_symbol(lib_id: str, project_lib: Path | None = None) -> list:
    """Return a self-contained symbol definition renamed to 'Lib:Name'."""
    lib, name = lib_id.split(":")
    path = project_lib if project_lib and project_lib.stem == lib else KICAD_SYMBOLS / f"{lib}.kicad_sym"
    syms = {unq(s[1]): s for s in find(load_lib(path), "symbol")}
    sym = [e if not isinstance(e, list) else e for e in syms[name]]
    ext = find(sym, "extends")
    if ext:  # derived symbol: take the parent's drawing/pins, keep own properties
        parent = syms[unq(ext[0][1])]
        own_props = {unq(p[1]): p for p in find(sym, "property")}
        merged = ["symbol", q(name)]
        for e in parent[2:]:
            if isinstance(e, list) and e[0] == "property" and unq(e[1]) in own_props:
                merged.append(own_props.pop(unq(e[1])))
            elif isinstance(e, list) and e[0] == "symbol":
                sub = list(e)
                sub[1] = q(unq(e[1]).replace(unq(ext[0][1]), name, 1))
                merged.append(sub)
            else:
                merged.append(e)
        merged += list(own_props.values())
        sym = merged
    sym = list(sym)
    sym[1] = q(lib_id)
    return sym


def pins(sym: list) -> dict[str, tuple[float, float, float]]:
    """pin number -> (x, y, angle) in symbol coordinates (y up)."""
    out = {}

    def walk(x):
        for e in x:
            if isinstance(e, list):
                if e and e[0] == "pin":
                    at = find(e, "at")[0]
                    num = unq(find(e, "number")[0][1])
                    out[num] = (float(at[1]), float(at[2]), float(at[3]) if len(at) > 3 else 0.0)
                else:
                    walk(e)
    walk(sym)
    return out


def pin_names(sym: list) -> dict[str, str]:
    out = {}

    def walk(x):
        for e in x:
            if isinstance(e, list):
                if e and e[0] == "pin":
                    out[unq(find(e, "number")[0][1])] = unq(find(e, "name")[0][1])
                else:
                    walk(e)
    walk(sym)
    return out


# --------------------------------------------------------------- schematic --
class Schematic:
    def __init__(self, title: str, project: str, paper: str = "A3", project_lib: Path | None = None):
        self.title, self.project, self.paper = title, project, paper
        self.root = uid()
        self.libsyms: dict[str, list] = {}
        self.items: list[str] = []
        self.project_lib = project_lib
        self.refs: dict[str, int] = {}
        self.texts: list[str] = []
        self.date = ""
        self.rev = ""
        self.comments: list[str] = []
        self.stubs: list[tuple] = []     # (x1, y1, x2, y2, net, ref) for the overlap check

    # -- helpers
    def _sym(self, lib_id):
        if lib_id not in self.libsyms:
            self.libsyms[lib_id] = get_symbol(lib_id, self.project_lib)
        return self.libsyms[lib_id]

    @staticmethod
    def _xf(px, py, x, y, rot):
        r = math.radians(rot)
        rx = px * math.cos(r) - py * math.sin(r)
        ry = px * math.sin(r) + py * math.cos(r)
        return round(x + rx, 2), round(y - ry, 2)

    def pin_pos(self, lib_id, num, x, y, rot=0):
        px, py, pa = pins(self._sym(lib_id))[num]
        sx, sy = self._xf(px, py, x, y, rot)
        return sx, sy, (pa + rot) % 360

    # -- primitives
    def wire(self, x1, y1, x2, y2):
        self.items.append(f'(wire (pts (xy {x1} {y1}) (xy {x2} {y2})) (stroke (width 0) (type default)) (uuid "{uid()}"))')

    def label(self, name, x, y, angle=0):
        just = "right" if angle in (180, 270) else "left"
        self.items.append(f'(label {q(name)} (at {x} {y} {angle}) (effects (font (size 1.27 1.27)) (justify {just} bottom)) (uuid "{uid()}"))')

    def no_connect(self, x, y):
        self.items.append(f'(no_connect (at {x} {y}) (uuid "{uid()}"))')

    def text(self, s, x, y, size=1.8, bold=False):
        b = " bold" if bold else ""
        self.items.append(f'(text {q(s)} (exclude_from_sim no) (at {x} {y} 0) (effects (font (size {size} {size}){b}) (justify left bottom)) (uuid "{uid()}"))')

    def rect(self, x1, y1, x2, y2, dash=True):
        t = "dash" if dash else "default"
        self.items.append(f'(rectangle (start {x1} {y1}) (end {x2} {y2}) (stroke (width 0.3) (type {t})) (fill (type none)) (uuid "{uid()}"))')

    def polyline(self, pts, dash=True):
        t = "dash" if dash else "default"
        xy = " ".join(f"(xy {a} {b})" for a, b in pts)
        self.items.append(f'(polyline (pts {xy}) (stroke (width 0.4) (type {t})) (uuid "{uid()}"))')

    def add(self, lib_id, ref_prefix, value, x, y, rot=0, nets=None, footprint="", stub=2.54,
            ref=None, fields=None, unit=1, hide_value=False, ref_at=None, val_at=None):
        """Place a symbol; nets maps pin number -> net name (None = no-connect)."""
        x, y = snap(x), snap(y)                 # keep every pin on KiCad's 1.27 mm grid
        sym = self._sym(lib_id)
        if ref is None:
            self.refs[ref_prefix] = self.refs.get(ref_prefix, 0) + 1
            ref = f"{ref_prefix}{self.refs[ref_prefix]}"
        pin_list = pins(sym)
        hv = " (hide yes)" if hide_value else ""
        # put the reference/value text outside the part, based on where its pins are
        pts = [self._xf(px, py, x, y, rot) for px, py, _ in pin_list.values()] or [(x, y)]
        ex = max(abs(px - x) for px, _ in pts)
        ey = max(abs(py - y) for _, py in pts)
        if ex <= 2.6:                                   # vertical two-pin part: text on the right
            (rx, ry), (vx, vy), just = (x + 3.3, y - 1.27), (x + 3.3, y + 1.27), "left"
        elif ey <= 2.6:                                 # horizontal two-pin part: text above / below
            (rx, ry), (vx, vy), just = (x, y - 2.54), (x, y + 3.3), "center"
        else:                                           # larger symbol: above and below its body
            (rx, ry), (vx, vy), just = (x - ex + 2.54, y - ey - 1.27), (x - ex + 2.54, y + ey + 2.54), "left"
        if ref_at:
            rx, ry = x + ref_at[0], y + ref_at[1]
        if val_at:
            vx, vy = x + val_at[0], y + val_at[1]
        jl = "" if just == "center" else f" (justify {just})"
        props = [
            # KiCad draws field text rotated with the symbol unless the field carries the same angle
            f'(property "Reference" {q(ref)} (at {rx} {ry} {rot % 180}) (effects (font (size 1.27 1.27)){jl}))',
            f'(property "Value" {q(value)} (at {vx} {vy} {rot % 180}) (effects (font (size 1.27 1.27)){jl}){hv})',
            f'(property "Footprint" {q(footprint)} (at {x} {y} 0) (effects (font (size 1.27 1.27)) (hide yes)))',
            f'(property "Datasheet" "" (at {x} {y} 0) (effects (font (size 1.27 1.27)) (hide yes)))',
        ]
        for k, v in (fields or {}).items():
            props.append(f'(property {q(k)} {q(v)} (at {x} {y} 0) (effects (font (size 1.27 1.27)) (hide yes)))')
        pin_txt = " ".join(f'(pin {q(n)} (uuid "{uid()}"))' for n in pin_list)
        self.items.append(
            f'(symbol (lib_id {q(lib_id)}) (at {x} {y} {rot}) (unit {unit}) (exclude_from_sim no) '
            f'(in_bom yes) (on_board yes) (dnp no) (uuid "{uid()}") ' + " ".join(props) + " " + pin_txt +
            f' (instances (project {q(self.project)} (path "/{self.root}" (reference {q(ref)}) (unit {unit})))))')
        for num, net in (nets or {}).items():
            sx, sy, ang = self.pin_pos(lib_id, num, x, y, rot)
            if net is None:
                self.no_connect(sx, sy)
                continue
            # stub points away from the symbol body (opposite to the pin's direction)
            dx = round(math.cos(math.radians(ang + 180)) * stub, 2)
            dy = round(-math.sin(math.radians(ang + 180)) * stub, 2)
            ex, ey = round(sx + dx, 2), round(sy + dy, 2)
            self.wire(sx, sy, ex, ey)
            self.stubs.append((sx, sy, ex, ey, net, ref))
            la = {0: 0, 90: 90, 180: 180, 270: 270}[int(round((ang + 180) % 360))]
            self.label(net, ex, ey, la)
        return ref

    # -- checks
    def overlaps(self):
        """Pairs of stubs of DIFFERENT nets that touch: those would short on the sheet."""
        def on_seg(px, py, s):
            x1, y1, x2, y2 = s[:4]
            return (min(x1, x2) - 0.01 <= px <= max(x1, x2) + 0.01 and
                    min(y1, y2) - 0.01 <= py <= max(y1, y2) + 0.01 and
                    abs((x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)) < 0.01)
        bad = []
        for i, a in enumerate(self.stubs):
            for b in self.stubs[i + 1:]:
                if a[4] == b[4]:
                    continue
                if any(on_seg(px, py, b) for px, py in ((a[0], a[1]), (a[2], a[3]))) or                    any(on_seg(px, py, a) for px, py in ((b[0], b[1]), (b[2], b[3]))):
                    bad.append((a[5], a[4], b[5], b[4], (a[0], a[1])))
        return bad

    # -- output
    def write(self, path: Path):
        bad = self.overlaps()
        if bad:
            raise RuntimeError("pin stubs of different nets touch: " + "; ".join(
                f"{ra}:{na} / {rb}:{nb} at {pt}" for ra, na, rb, nb, pt in bad))
        tb = [f"(title {q(self.title)})"]
        if self.date:
            tb.append(f"(date {q(self.date)})")
        if self.rev:
            tb.append(f"(rev {q(self.rev)})")
        for k, c in enumerate(self.comments, 1):
            tb.append(f"(comment {k} {q(c)})")
        libs = "\n".join(dump(s, 2) for s in self.libsyms.values())
        body = "\n\t".join(self.items)
        text = (f'(kicad_sch\n\t(version 20250114)\n\t(generator "eeschema")\n\t(generator_version "9.0")\n'
                f'\t(uuid "{self.root}")\n\t(paper "{self.paper}")\n\t(title_block {" ".join(tb)})\n'
                f'\t(lib_symbols\n{libs}\n\t)\n\t{body}\n'
                f'\t(sheet_instances (path "/" (page "1")))\n\t(embedded_fonts no)\n)\n')
        path.write_text(text, encoding="utf-8")
