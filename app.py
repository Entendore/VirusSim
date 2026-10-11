"""
Agent-based multi-strain epidemic simulation (SEIR + deaths) with YouTube
Shorts export (single / batch sweep / head-to-head split-screen), conquest
map, live interventions, Rt tracking, and a procedural soundtrack.

Improvements in this version:
  - Perf: exact spatial-grid contact search (O(n log n), not O(n²)) —
    2000 agents stay smooth; cached visual state.
  - Live interventions: vaccinate-now, timed lockdown (L), isolation
    quarantine toggle. Isolation also applies inside rendered videos.
  - Live Rt estimate (per strain in showdown) in the stats bar and burned
    into video footers.
  - Video polish: title fade-in, Shorts-style duration bar, alert chips.
  - Optional procedural soundtrack whose intensity tracks the outbreak,
    muxed with the bundled ffmpeg.
  - CSV export of the epidemic curve. Shortcuts: Space, L, V.

Requires:
    pip install PySide6 numpy imageio imageio-ffmpeg
Run:
    python virus_simulation.py
"""

import csv
import importlib.util
import os
import subprocess
import sys
from datetime import datetime

import numpy as np
from PySide6.QtCore import (
    QStandardPaths, Qt, QRectF, QTimer, QThread, QUrl, Signal, Slot,
)
from PySide6.QtGui import (
    QColor, QDesktopServices, QFont, QImage, QPainter, QPainterPath, QPen,
    QPointF, QPixmap,
)
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea,
    QSizePolicy, QSlider, QSplitter, QVBoxLayout, QWidget,
)

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
SUSCEPTIBLE, EXPOSED, INFECTED, RECOVERED, DEAD = 0, 1, 2, 3, 4
TICKS_PER_DAY = 20
WORLD_W, WORLD_H = 1000, 600
MAX_SHORT_SECONDS = 180
SUPER_SPREAD_FACTOR = 3.0
OVERLOAD_MORTALITY_MULT = 4.0
MAP_COLS, MAP_ROWS = 40, 24

STATE_COLORS = {
    SUSCEPTIBLE: QColor("#4da3ff"),
    EXPOSED:     QColor("#f0883e"),
    INFECTED:    QColor("#ff5c5c"),
    RECOVERED:   QColor("#3fb950"),
    DEAD:        QColor("#8b949e"),
}
EXPOSED_COLOR = STATE_COLORS[EXPOSED]
AGENT_RADIUS = {SUSCEPTIBLE: 2.6, EXPOSED: 2.8, INFECTED: 3.2,
                RECOVERED: 2.6, DEAD: 2.2}

STRAIN_NAMES = ("Alpha", "Beta", "Gamma")
STRAIN_GREEK = ("α", "β", "γ")
STRAIN_COLORS = (QColor("#ff5c5c"), QColor("#bc8cff"), QColor("#f0883e"))
H2H_COLORS = (QColor("#58a6ff"), QColor("#f778ba"))

IMAGEIO_OK = importlib.util.find_spec("imageio") is not None

PRESETS = {
    "Custom": dict(chance=10, radius=9, incubation=2, recovery=25,
                   mortality=80, superspreader=4, waning=0,
                   desc="your own slider settings"),
    "Seasonal Flu": dict(chance=6, radius=9, incubation=2, recovery=8,
                         mortality=5, superspreader=4, waning=180,
                         desc="mild but relentless — low deaths, immunity fades each season"),
    "Common Cold": dict(chance=8, radius=8, incubation=1, recovery=5,
                        mortality=0, superspreader=2, waning=45,
                        desc="harmless, fast, and always coming back"),
    "COVID-19-like": dict(chance=8, radius=10, incubation=4, recovery=14,
                          mortality=20, superspreader=10, waning=0,
                          desc="silent incubation + super-spreaders overwhelm the hospitals"),
    "Measles-like": dict(chance=30, radius=14, incubation=7, recovery=10,
                         mortality=10, superspreader=8, waning=0,
                         desc="one of the most contagious viruses known"),
    "Smallpox-like": dict(chance=18, radius=8, incubation=10, recovery=16,
                          mortality=300, superspreader=6, waning=0,
                          desc="slow, visible, and very deadly"),
    "Ebola-like": dict(chance=12, radius=6, incubation=8, recovery=12,
                       mortality=500, superspreader=5, waning=0,
                       desc="short, brutal outbreaks with very high mortality"),
    "Zombie Plague": dict(chance=35, radius=9, incubation=0, recovery=0,
                          mortality=0, superspreader=0, waning=0,
                          desc="recovery 0 = never recover — everyone turns"),
}

MATCHUPS = {
    "Custom": None,
    "Flu vs COVID-19": dict(n=2, cross=0, strains=[
        dict(chance=6, inc=2, rec=8, mort=5),
        dict(chance=8, inc=4, rec=14, mort=20)]),
    "Measles vs Ebola": dict(n=2, cross=0, strains=[
        dict(chance=30, inc=7, rec=10, mort=10),
        dict(chance=12, inc=8, rec=12, mort=500)]),
    "Zombie vs Smallpox": dict(n=2, cross=0, strains=[
        dict(chance=25, inc=0, rec=0, mort=0),
        dict(chance=18, inc=10, rec=16, mort=300)]),
    "Variant vs Original (80% cross-immunity)": dict(n=2, cross=80, strains=[
        dict(chance=10, inc=3, rec=14, mort=15),
        dict(chance=13, inc=2, rec=12, mort=10)]),
    "Three-way: Flu · COVID · Zombie": dict(n=3, cross=0, strains=[
        dict(chance=8, inc=2, rec=10, mort=5),
        dict(chance=14, inc=3, rec=18, mort=15),
        dict(chance=22, inc=0, rec=0, mort=0)]),
}

STYLESHEET = """
QMainWindow, QWidget#root { background-color: #0f141a; }
QLabel { color: #c9d1d9; background: transparent; }
QLabel#hint { color: #6e7681; font-size: 11px; }
QLabel#status { color: #58a6ff; font-weight: bold; }
QLabel#statValue { font-size: 20px; font-weight: 700; }
QLabel#sliderValue { color: #8b949e; font-size: 12px; }

QGroupBox {
    border: 1px solid #21262d; border-radius: 10px;
    margin-top: 16px; padding: 6px 8px 8px 8px;
    font-size: 12px; font-weight: bold; color: #8b949e;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QGroupBox#strain0::title { color: #ff5c5c; }
QGroupBox#strain1::title { color: #bc8cff; }
QGroupBox#strain2::title { color: #f0883e; }

QSlider { min-height: 20px; }
QSlider::groove:horizontal { height: 4px; background: #21262d; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #1f6feb; border-radius: 2px; }
QSlider::handle:horizontal { width: 14px; margin: -5px 0; background: #e6edf3; border-radius: 7px; }
QSlider::handle:horizontal:hover { background: #ffffff; }

QPushButton {
    background: #21262d; color: #c9d1d9; border: 1px solid #30363d;
    border-radius: 6px; padding: 8px 12px; font-weight: 600;
}
QPushButton:hover { background: #2a3038; }
QPushButton#runBtn { background: #238636; border-color: #238636; color: white; }
QPushButton#runBtn:hover { background: #2ea043; }
QPushButton#shortBtn { background: #1f6feb; border-color: #1f6feb; color: white; }
QPushButton#shortBtn:hover { background: #388bfd; }
QPushButton#batchBtn { background: #8957e5; border-color: #8957e5; color: white; }
QPushButton#batchBtn:hover { background: #a371f7; }
QPushButton#h2hBtn { background: #db6d28; border-color: #db6d28; color: white; }
QPushButton#h2hBtn:hover { background: #f0883e; }
QPushButton:disabled { color: #6e7681; }

QLineEdit, QComboBox, QSpinBox {
    background: #161b22; color: #c9d1d9; border: 1px solid #30363d;
    border-radius: 6px; padding: 6px 9px;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus { border-color: #1f6feb; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox::down-arrow {
    image: none; border-left: 4px solid transparent;
    border-right: 4px solid transparent; border-top: 6px solid #8b949e;
    margin-right: 8px;
}
QComboBox QAbstractItemView {
    background: #161b22; color: #c9d1d9;
    selection-background-color: #1f6feb; border: 1px solid #30363d;
}
QSpinBox::up-button, QSpinBox::down-button { width: 16px; }
QSpinBox::up-arrow { image: none; border-left: 4px solid transparent; border-right: 4px solid transparent; border-bottom: 5px solid #8b949e; }
QSpinBox::down-arrow { image: none; border-left: 4px solid transparent; border-right: 4px solid transparent; border-top: 5px solid #8b949e; }
QSpinBox:disabled { color: #6e7681; }

QListWidget {
    background: #0d1117; color: #c9d1d9; border: 1px solid #30363d;
    border-radius: 6px; padding: 4px;
}
QCheckBox { color: #c9d1d9; }
QCheckBox::indicator {
    width: 15px; height: 15px; border: 1px solid #30363d;
    border-radius: 4px; background: #161b22;
}
QCheckBox::indicator:checked { background: #1f6feb; border-color: #1f6feb; }

QProgressBar {
    background: #161b22; border: 1px solid #30363d; border-radius: 6px;
    height: 16px; text-align: center; color: #c9d1d9; font-size: 10px;
}
QProgressBar::chunk { background: #238636; border-radius: 5px; }

QScrollArea { background: transparent; border: none; }
QScrollArea > QWidget > QWidget { background: transparent; }
QScrollBar:vertical { background: #0f141a; width: 10px; }
QScrollBar::handle:vertical { background: #30363d; border-radius: 5px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

QSplitter::handle { background: #0f141a; width: 6px; }
"""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _px_font(px, bold=False):
    f = QFont()
    f.setPixelSize(max(6, int(round(px))))
    if bold:
        f.setBold(True)
    return f


def _dim(color, factor=0.6):
    return QColor(int(color.red() * factor),
                  int(color.green() * factor),
                  int(color.blue() * factor))


def _alpha(color, a):
    c = QColor(color)
    c.setAlpha(a)
    return c


def qimage_to_array(img):
    h, w = img.height(), img.width()
    bpl = img.bytesPerLine()
    arr = np.frombuffer(img.bits(), dtype=np.uint8, count=h * bpl).reshape(h, bpl)
    return arr[:, :w * 3].reshape(h, w, 3).copy()


def _videos_dir():
    try:
        folder = QStandardPaths.writableLocation(QStandardPaths.MoviesLocation)
        if folder:
            return folder
    except Exception:
        pass
    home = os.path.join(os.path.expanduser("~"), "Videos")
    if os.path.isdir(home) and os.access(home, os.W_OK):
        return home
    return "."


def _slugify(text):
    keep = "".join(c if c.isalnum() else "-" for c in text.lower())
    parts = [p for p in keep.split("-") if p]
    return "-".join(parts)[:40] or "variant"


def shorts_path(stamp, label, index, total):
    name = (f"virus_short_{stamp}_{index:02d}of{total:02d}_"
            f"{_slugify(label or 'outbreak')}.mp4")
    return os.path.join(_videos_dir(), name)


def _rt_text(rt, showdown):
    """Format an Rt array for display."""
    if rt is None:
        return None
    vals = []
    for x in np.ravel(rt):
        if np.isnan(x):
            vals.append("–")
        elif np.isinf(x):
            vals.append("9.9+")
        else:
            vals.append(f"{x:.2f}")
    if showdown and len(vals) > 1:
        return "Rt " + " ".join(f"{g}{v}" for g, v in zip(STRAIN_GREEK, vals))
    return f"Rt {vals[0]}" if vals else None


def _active_curve(sim, frames, fps, tpf):
    """Per-rendered-frame active infection fraction (E+I)/pop for the music."""
    hist = sim.history
    n = hist.shape[0]
    pop = max(sim.n_agents, 1)
    idx = np.minimum(np.arange(max(frames, 1)) * max(tpf, 1), n - 1)
    return (hist[idx, 2] + hist[idx, 3]) / pop


def _generate_soundtrack(wav_path, act, fps):
    """Simple procedural chiptune whose intensity follows the outbreak."""
    import wave
    act = np.asarray(act, dtype=np.float64)
    frames = act.size
    if frames <= 0:
        raise RuntimeError("no frames for soundtrack")
    sr = 44100
    total = int(frames / fps * sr)
    rng = np.random.default_rng(20240501)
    buf = np.zeros(total + sr, np.float64)      # tail padding for overshoot

    def add(pos, sig, vol):
        if sig.size == 0 or vol <= 0:
            return
        end = min(buf.size, pos + sig.size)
        if end > pos:
            buf[pos:end] += sig[:end - pos] * vol

    tt = np.arange(int(sr * 0.30)) / sr
    kick = np.sin(2 * np.pi * (50 * tt + 28 * tt ** 2)) * np.exp(-tt * 16)
    hat = rng.standard_normal(int(sr * 0.05)) * \
        np.exp(-np.arange(int(sr * 0.05)) / sr * 90)
    sn = np.arange(int(sr * 0.18)) / sr
    snare = (rng.standard_normal(sn.size) * 0.8 +
             np.sin(2 * np.pi * 185 * sn) * 0.5) * np.exp(-sn * 24)
    notes = (55.0, 65.41, 73.42, 82.41)

    eighth = int(sr * 0.25)                     # 120 BPM 8th notes
    k = 0
    for pos in range(0, total, eighth):
        f = min(int(pos / sr * fps), frames - 1)
        it = float(np.clip(act[f], 0.0, 1.0))
        if k % 2 == 0:
            add(pos, kick, 0.42 + 0.20 * it)
        elif it > 0.15:
            add(pos, hat, 0.06 + 0.20 * it)
        if k % 8 == 4 and it > 0.30:
            add(pos, snare, 0.16 + 0.14 * it)
        if it > 0.05:
            base = notes[(k // 2) % 4] * (2.0 ** (0.2 * it))
            t2 = np.arange(eighth) / sr
            tone = (np.sign(np.sin(2 * np.pi * base * t2)) * 0.35 +
                    np.sin(2 * np.pi * base * 2 * t2) * 0.25) * np.exp(-t2 * 7)
            add(pos, tone, 0.09 + 0.15 * it)
        k += 1

    # tension drone whose volume follows the smoothed outbreak intensity
    sm = np.convolve(act, np.ones(max(1, fps // 2)) / max(1, fps // 2),
                     mode="same")
    amp = np.repeat(0.035 + 0.11 * sm, int(sr / fps))[:total]
    t_all = np.arange(total) / sr
    fm = 108.0 * (1.0 + 0.05 * np.sin(2 * np.pi * 0.13 * t_all))
    buf[:total] += np.sin(2 * np.pi * np.cumsum(fm) / sr) * amp

    buf = buf[:total]
    peak = float(np.max(np.abs(buf))) or 1.0
    buf = buf / peak * 0.85
    fade = min(buf.size, int(sr * 1.0))
    if fade > 0:
        buf[-fade:] *= np.linspace(1.0, 0.0, fade)
    data = (buf * 32767.0).astype("<i2")
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.tobytes())


def _mux_audio(video_path, wav_path):
    from imageio_ffmpeg import get_ffmpeg_exe
    exe = get_ffmpeg_exe()
    tmp = video_path + ".mux.mp4"
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.run(
        [exe, "-y", "-i", video_path, "-i", wav_path,
         "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
         "-movflags", "+faststart", "-shortest", tmp],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kw)
    os.replace(tmp, video_path)


def _attach_soundtrack(path, act, fps):
    wav = path + ".wav"
    _generate_soundtrack(wav, act, fps)
    _mux_audio(path, wav)
    os.remove(wav)


def draw_conquest_map(p, sim, rect, colors):
    dom, intensity = sim.map_data()
    rows, cols = dom.shape
    cw, ch = rect.width() / cols, rect.height() / rows
    for r in range(rows):
        base_y = rect.top() + r * ch
        for c in range(cols):
            a = float(intensity[r, c])
            if a < 0.02:
                continue
            col = QColor(colors[int(dom[r, c])])
            col.setAlphaF(min(0.60, 0.10 + 0.55 * a))
            p.fillRect(QRectF(rect.left() + c * cw, base_y, cw + 0.6, ch + 0.6), col)


def renderer_meta(s):
    strains_meta = []
    for i, sp in enumerate(s["strains"]):
        name = STRAIN_NAMES[i] if s["showdown"] else "Virus"
        strains_meta.append(dict(
            name=name, chance=sp["chance"],
            incubation_days=int(round(sp["incubation_ticks"] / TICKS_PER_DAY)),
            recovery_days=int(round(sp["recovery_ticks"] / TICKS_PER_DAY)),
            mortality=sp["mortality"],
        ))
    return dict(
        population=s["population"], radius=s["radius"],
        hospital_frac=s.get("hospital_frac", 0.15),
        showdown=s["showdown"], n_strains=len(s["strains"]),
        strains=strains_meta, label=s.get("label"),
        cross_immunity=s.get("cross_immunity", 0.0),
        show_map=bool(s.get("show_map", False)),
        isolate=bool(s.get("isolate", False)),
    )


def side_paramline(cfg):
    sp = cfg["strains"][0]
    rec = int(round(sp["recovery_ticks"] / TICKS_PER_DAY))
    parts = [f"transmit {sp['chance'] * 100:.4g}%",
             f"incub {int(round(sp['incubation_ticks'] / TICKS_PER_DAY))} d",
             "recovery never" if rec == 0 else f"recovery {rec} d",
             f"mort {sp['mortality'] * 100:.4g}%"]
    if cfg.get("distancing", 0) > 0:
        parts.append(f"distancing {cfg['distancing'] * 100:.0f}%")
    if cfg.get("vaccination", 0) > 0:
        parts.append(f"vacc {cfg['vaccination'] * 100:.0f}%")
    if cfg.get("super_fraction", 0) > 0:
        parts.append(f"super-spreaders {cfg['super_fraction'] * 100:.0f}%")
    if cfg.get("isolate"):
        parts.append("isolation ON")
    return " · ".join(parts)


# ---------------------------------------------------------------------------
# Simulation engine (numpy-vectorised, multi-strain SEIR + conquest map)
# ---------------------------------------------------------------------------
class Simulation:
    """Agent-based epidemic model. 1 strain = classic mode; 2-3 = showdown."""

    def __init__(self, width=WORLD_W, height=WORLD_H, seed=None):
        self.width, self.height = width, height
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.peak_infected = 0
        self.n_strains = 1
        self.cross_immunity = 0.0
        self.strains = [dict(chance=0.10, incubation_ticks=2 * TICKS_PER_DAY,
                             recovery_ticks=25 * TICKS_PER_DAY, mortality=0.08)]
        self.isolate_infected = False
        self.lockdown_until = -1
        # visual-state cache (bumped on every mutation)
        self._epoch = 0
        self._vis_epoch = -1
        self._vis = (None, None)
        # new-infection tracking for Rt
        self._new = np.zeros((4096, 3), np.float64)
        self._new_len = 0
        self._pending_new = np.zeros(3, np.float64)
        self.pos = np.zeros((0, 2)); self.vel = np.zeros((0, 2))
        self.dir = np.zeros((0, 2)); self.moving = np.zeros(0, bool)
        self.strain_state = np.zeros((1, 0), np.int8)
        self.infection_clock = np.zeros((1, 0), np.int64)
        self.dead = np.zeros(0, bool)
        self.ever_infected_mask = np.zeros(0, bool)
        self.ever_per_strain = np.zeros(1, np.int64)
        self.super_spreader = np.zeros(0, bool)
        self.cell_infections = np.zeros((3, MAP_ROWS, MAP_COLS), np.int64)
        self.conquest = np.full((MAP_ROWS, MAP_COLS), -1, np.int8)
        self._init_history()

    # -- basic properties ----------------------------------------------------
    @property
    def n_agents(self):
        return int(self.dead.size)

    @property
    def showdown(self):
        return self.n_strains > 1

    @property
    def ever_infected(self):
        return int(self.ever_infected_mask.sum())

    @property
    def in_lockdown(self):
        return self.lockdown_until > self.step_count

    @property
    def lockdown_remaining_days(self):
        return max(0.0, (self.lockdown_until - self.step_count) / TICKS_PER_DAY)

    def set_lockdown(self, days):
        self.lockdown_until = self.step_count + int(days * TICKS_PER_DAY)

    def clear_lockdown(self):
        self.lockdown_until = -1

    def _bump(self):
        self._epoch += 1

    # -- history (t, S, E, I, R, D, I_alpha, I_beta, I_gamma) -----------------
    def _init_history(self):
        self._hist = np.zeros((4096, 9), np.int64)
        self._hist_len = 0

    @property
    def history(self):
        return self._hist[:self._hist_len]

    def _record(self):
        if self._hist_len >= self._hist.shape[0]:
            grown = np.zeros((self._hist.shape[0] * 2, 9), np.int64)
            grown[:self._hist_len] = self._hist[:self._hist_len]
            self._hist = grown
        alive = ~self.dead
        ss = self.strain_state
        anyI = (ss == INFECTED).any(axis=0) & alive
        anyE = (ss == EXPOSED).any(axis=0) & alive
        allS = (ss == SUSCEPTIBLE).all(axis=0) & alive
        S, E, I = int(allS.sum()), int(anyE.sum()), int(anyI.sum())
        D = int(self.dead.sum())
        R = int(alive.sum()) - S - E - I
        row = [self.step_count, S, E, I, R, D]
        for s in range(3):
            row.append(int((alive & (ss[s] == INFECTED)).sum())
                       if s < self.n_strains else 0)
        self._hist[self._hist_len] = row
        self._hist_len += 1
        if I > self.peak_infected:
            self.peak_infected = I
        # new-infection bookkeeping (for Rt)
        if self._new_len >= self._new.shape[0]:
            grown = np.zeros((self._new.shape[0] * 2, 3), np.float64)
            grown[:self._new_len] = self._new[:self._new_len]
            self._new = grown
        self._new[self._new_len] = self._pending_new
        self._new_len += 1
        self._pending_new[:] = 0.0

    def estimate_rt(self, strains, window_days=None):
        """Ratio estimator: new infections now vs one serial interval ago."""
        if self._new_len < 2:
            return np.full(self.n_strains, np.nan)
        if window_days is None:
            sp = strains[0] if strains else self.strains[0]
            d = (sp["incubation_ticks"] + 0.5 * sp["recovery_ticks"]) / TICKS_PER_DAY
            window_days = max(2.0, d)
        w = max(1, int(window_days * TICKS_PER_DAY))
        n = self._new_len
        if n < 2 * w:
            return np.full(self.n_strains, np.nan)
        cur = self._new[n - w:n, :self.n_strains].sum(axis=0)
        prev = self._new[n - 2 * w:n - w, :self.n_strains].sum(axis=0)
        rt = np.zeros(self.n_strains)
        m = prev > 0
        rt[m] = cur[m] / prev[m]
        rt[(prev == 0) & (cur > 0)] = np.inf
        return rt

    def visual(self):
        """Per-agent display state + which strain caused it (-1 if n/a). Cached."""
        if self._vis_epoch == self._epoch:
            return self._vis
        n = self.n_agents
        vis = np.full(n, SUSCEPTIBLE, np.int8)
        vstr = np.full(n, -1, np.int8)
        vis[self.dead] = DEAD
        rem = ~self.dead
        ss = self.strain_state
        for s in range(self.n_strains):
            m = rem & (ss[s] == INFECTED)
            if m.any():
                vis[m] = INFECTED; vstr[m] = s; rem &= ~m
        for s in range(self.n_strains):
            m = rem & (ss[s] == EXPOSED)
            if m.any():
                vis[m] = EXPOSED; vstr[m] = s; rem &= ~m
        m = rem & (ss == RECOVERED).any(axis=0)
        if m.any():
            vis[m] = RECOVERED
        self._vis = (vis, vstr)
        self._vis_epoch = self._epoch
        return self._vis

    def count_blocks(self):
        vis, vstr = self.visual()
        S = int((vis == SUSCEPTIBLE).sum()); E = int((vis == EXPOSED).sum())
        I = int((vis == INFECTED).sum());    R = int((vis == RECOVERED).sum())
        D = int((vis == DEAD).sum())
        sick = []
        for s in range(self.n_strains):
            sick.append(int(((vis == INFECTED) & (vstr == s)).sum()
                            + ((vis == EXPOSED) & (vstr == s)).sum()))
        return S, E, I, R, D, sick

    def map_data(self):
        total = self.cell_infections[:self.n_strains].sum(axis=0)
        dom = np.argmax(self.cell_infections[:self.n_strains], axis=0).astype(np.int8)
        dom[total == 0] = -1
        denom = max(1.0, self.n_agents / (MAP_ROWS * MAP_COLS))
        intensity = np.clip(total / denom, 0.0, 1.0)
        return dom, intensity

    def _record_infection_cells(self, s, idx):
        if idx.size == 0 or s >= self.n_strains:
            return
        cx = np.clip((self.pos[idx, 0] * (MAP_COLS / self.width)).astype(np.int64),
                     0, MAP_COLS - 1)
        cy = np.clip((self.pos[idx, 1] * (MAP_ROWS / self.height)).astype(np.int64),
                     0, MAP_ROWS - 1)
        np.add.at(self.cell_infections[s], (cy, cx), 1)
        flat = cy * MAP_COLS + cx
        unc = flat[self.conquest.ravel()[flat] < 0]
        if unc.size:
            self.conquest.ravel()[unc] = s

    # -- setup ---------------------------------------------------------------
    def reset(self, population=700, initial_infected=5, speed=3.0,
              distancing=0.0, vaccination=0.0, super_fraction=0.0,
              strains=None, cross_immunity=0.0):
        if not strains:
            strains = [dict(chance=0.10, incubation_ticks=2 * TICKS_PER_DAY,
                            recovery_ticks=25 * TICKS_PER_DAY, mortality=0.08)]
        self.strains = [dict(s) for s in strains]
        self.n_strains = len(self.strains)
        self.cross_immunity = float(cross_immunity)
        self.lockdown_until = -1

        n = int(population)
        rng = self.rng
        self.pos = np.column_stack([rng.uniform(0, self.width, n),
                                    rng.uniform(0, self.height, n)])
        ang = rng.uniform(0, 2 * np.pi, n)
        self.dir = np.column_stack([np.cos(ang), np.sin(ang)])
        self.moving = rng.random(n) >= distancing
        self.vel = self.dir * speed * self.moving[:, None]
        self.super_spreader = rng.random(n) < super_fraction
        self.strain_state = np.full((self.n_strains, n), SUSCEPTIBLE, np.int8)
        self.infection_clock = np.zeros((self.n_strains, n), np.int64)
        self.dead = np.zeros(n, bool)
        self.ever_infected_mask = np.zeros(n, bool)
        self.ever_per_strain = np.zeros(self.n_strains, np.int64)
        self.cell_infections[:] = 0
        self.conquest.fill(-1)
        self._new_len = 0
        self._pending_new[:] = 0.0

        if vaccination > 0:
            k = min(n, int(round(n * vaccination)))
            if k:
                idx = rng.choice(n, size=k, replace=False)
                self.strain_state[:, idx] = RECOVERED

        sus0 = np.flatnonzero(self.strain_state[0] == SUSCEPTIBLE)
        per = max(1, int(initial_infected))
        if self.n_strains == 1:
            m = min(per, sus0.size)
            if m:
                first = rng.choice(sus0, size=m, replace=False)
                self.strain_state[0, first] = INFECTED
                self.ever_per_strain[0] = m
                self.ever_infected_mask[first] = True
        else:
            m = min(per, sus0.size // self.n_strains)
            if m:
                perm = rng.permutation(sus0.size)
                for s in range(self.n_strains):
                    ids = sus0[perm[s * m:(s + 1) * m]]
                    self.strain_state[s, ids] = INFECTED
                    self.ever_per_strain[s] = ids.size
                    self.ever_infected_mask[ids] = True

        self.step_count = 0
        self.peak_infected = int(((self.strain_state == INFECTED).any(axis=0)
                                  & ~self.dead).sum())
        self._init_history()
        self._bump()
        self._record()

    def set_speed(self, speed):
        if self.n_agents:
            self.vel = self.dir * speed * self.moving[:, None]

    def vaccinate_now(self, fraction):
        """Immediately immunise a fraction of fully-susceptible living agents."""
        if fraction <= 0 or self.n_agents == 0:
            return 0
        cand = (~self.dead) & \
            (self.strain_state == SUSCEPTIBLE).all(axis=0)
        idx = np.flatnonzero(cand)
        if idx.size == 0:
            return 0
        k = min(idx.size, int(round(idx.size * fraction)))
        sel = self.rng.choice(idx, size=k, replace=False)
        self.strain_state[:, sel] = RECOVERED
        self.infection_clock[:, sel] = 0
        self._bump()
        self._record()
        return k

    def infect_at(self, x, y, radius=16.0, max_count=30, strain=0):
        if self.n_agents == 0:
            return 0
        s = min(max(int(strain), 0), self.n_strains - 1)
        d2 = (self.pos[:, 0] - x) ** 2 + (self.pos[:, 1] - y) ** 2
        near = np.flatnonzero((d2 <= radius * radius) & (~self.dead)
                              & (self.strain_state[s] == SUSCEPTIBLE))
        if near.size == 0:
            return 0
        if near.size > max_count:
            near = self.rng.choice(near, max_count, replace=False)
        self.strain_state[s, near] = EXPOSED
        self.infection_clock[s, near] = 0
        fresh = near[~self.ever_infected_mask[near]]
        self.ever_infected_mask[fresh] = True
        self.ever_per_strain[s] += near.size
        self._pending_new[s] += near.size
        self._record_infection_cells(s, near)
        self._bump()
        self._record()
        return int(near.size)

    # -- one tick --------------------------------------------------------------
    def step(self, radius, strains, waning_ticks=0, hospital_cap=0):
        params = list(strains) if len(strains) == self.n_strains else self.strains
        self._move()
        for s in range(self.n_strains):
            self._transmit_strain(s, radius, params[s])
        self._progress(params, waning_ticks, hospital_cap)
        self.step_count += 1
        self._bump()
        self._record()
        return tuple(int(v) for v in self._hist[self._hist_len - 1])

    def _move(self):
        moving = self.moving
        if self.in_lockdown:
            moving = np.zeros_like(moving)
        elif self.isolate_infected:
            iso = (self.strain_state == INFECTED).any(axis=0)
            if iso.any():
                moving = moving & ~iso
        self.pos += self.vel * moving[:, None]
        x, y = self.pos[:, 0], self.pos[:, 1]
        flip_x = (x <= 0) | (x >= self.width)
        flip_y = (y <= 0) | (y >= self.height)
        self.vel[flip_x, 0] *= -1
        self.vel[flip_y, 1] *= -1
        np.clip(x, 0, self.width, out=x)
        np.clip(y, 0, self.height, out=y)

    def _count_within(self, targets, sources, radius):
        """Exact count of sources within radius of each target.

        Spatial-hash acceleration: enumerate candidate pairs via grid cells
        (cell size = radius ⇒ 3×3 neighbourhood covers everything), then
        verify true distance per pair. Results identical to the old O(n²)
        matrix method, but ~O(n log n).
        """
        if targets.size == 0 or sources.size == 0 or radius <= 0:
            return np.zeros(targets.size, np.int64)
        cs = float(radius)
        gw = max(1, int(self.width / cs) + 1)
        gh = max(1, int(self.height / cs) + 1)
        scx = np.minimum((self.pos[sources, 0] / cs).astype(np.int64), gw - 1)
        scy = np.minimum((self.pos[sources, 1] / cs).astype(np.int64), gh - 1)
        skey = scy * gw + scx
        order = np.argsort(skey, kind="stable")
        skey = skey[order]
        sids = sources[order]
        tcx = np.minimum((self.pos[targets, 0] / cs).astype(np.int64), gw - 1)
        tcy = np.minimum((self.pos[targets, 1] / cs).astype(np.int64), gh - 1)
        out = np.zeros(targets.size, np.int64)
        r2 = radius * radius
        for dy in (-1, 0, 1):
            ny = tcy + dy
            oky = (ny >= 0) & (ny < gh)
            for dx in (-1, 0, 1):
                nx = tcx + dx
                ok = oky & (nx >= 0) & (nx < gw)
                if not ok.any():
                    continue
                nk = np.where(ok, np.clip(ny, 0, gh - 1) * gw +
                              np.clip(nx, 0, gw - 1), -1)
                lo = np.searchsorted(skey, nk, side="left")
                hi = np.searchsorted(skey, nk, side="right")
                cnt = hi - lo
                m = ok & (cnt > 0)
                if not m.any():
                    continue
                pos_idx = np.flatnonzero(m)
                c = cnt[m]
                l = lo[m]
                total = int(c.sum())
                rep = np.repeat(np.arange(pos_idx.size), c)
                starts = np.cumsum(c) - c
                within = (np.arange(total) - np.repeat(starts, c)
                          + np.repeat(l, c))
                src = sids[within]
                tgt = targets[pos_idx[rep]]
                d2 = ((self.pos[src] - self.pos[tgt]) ** 2).sum(axis=1)
                good = d2 <= r2
                if good.any():
                    np.add.at(out, pos_idx[rep[good]], 1)
        return out

    def _transmit_strain(self, s, radius, sp):
        chance = sp["chance"]
        if chance <= 0:
            return
        alive = ~self.dead
        sick = np.flatnonzero(alive & (self.strain_state[s] == INFECTED))
        target = np.flatnonzero(alive & (self.strain_state[s] == SUSCEPTIBLE))
        if sick.size == 0 or target.size == 0:
            return
        sup = self.super_spreader[sick]
        contacts = (self._count_within(target, sick[~sup], radius)
                    + self._count_within(target, sick[sup],
                                         radius * SUPER_SPREAD_FACTOR))
        p = 1.0 - (1.0 - chance) ** contacts
        newly = target[self.rng.random(target.size) < p]
        if newly.size:
            self.strain_state[s, newly] = EXPOSED
            self.infection_clock[s, newly] = 0
            fresh = newly[~self.ever_infected_mask[newly]]
            self.ever_infected_mask[fresh] = True
            self.ever_per_strain[s] += newly.size
            self._pending_new[s] += newly.size
            self._record_infection_cells(s, newly)

    def _progress(self, params, waning_ticks, hospital_cap):
        alive = ~self.dead
        ss = self.strain_state

        if waning_ticks > 0 and self.n_strains == 1:
            rec = np.flatnonzero(alive & (ss[0] == RECOVERED))
            if rec.size:
                self.infection_clock[0, rec] += 1
                expired = rec[self.infection_clock[0, rec] >= waning_ticks]
                if expired.size:
                    ss[0, expired] = SUSCEPTIBLE
                    self.infection_clock[0, expired] = 0

        for s in range(self.n_strains):
            inc = params[s]["incubation_ticks"]
            exp = np.flatnonzero(alive & (ss[s] == EXPOSED))
            if exp.size:
                self.infection_clock[s, exp] += 1
                matured = exp[self.infection_clock[s, exp] >= max(inc, 0)]
                if matured.size:
                    ss[s, matured] = INFECTED
                    self.infection_clock[s, matured] = 0

        total_I = int((alive & (ss == INFECTED).any(axis=0)).sum())
        overload = hospital_cap > 0 and total_I > hospital_cap

        for s in range(self.n_strains):
            rec_t = params[s]["recovery_ticks"]
            if rec_t <= 0:
                continue
            sick = np.flatnonzero(alive & (ss[s] == INFECTED))
            if sick.size == 0:
                continue
            self.infection_clock[s, sick] += 1
            done = sick[self.infection_clock[s, sick] >= rec_t]
            if done.size == 0:
                continue
            eff_m = params[s]["mortality"]
            if overload:
                eff_m = min(1.0, eff_m * OVERLOAD_MORTALITY_MULT)
            dies_mask = self.rng.random(done.size) < eff_m
            dies = done[dies_mask]
            survivors = done[~dies_mask]
            if dies.size:
                self.dead[dies] = True
                self.vel[dies] = 0.0
                self.moving[dies] = False
            if survivors.size:
                ss[s, survivors] = RECOVERED
                self.infection_clock[s, survivors] = 0
                ci = self.cross_immunity
                if ci > 0:
                    for t in range(self.n_strains):
                        if t == s:
                            continue
                        cand = survivors[ss[t, survivors] == SUSCEPTIBLE]
                        if cand.size:
                            grant = cand[self.rng.random(cand.size) < ci]
                            ss[t, grant] = RECOVERED
                            self.infection_clock[t, grant] = 0


# ---------------------------------------------------------------------------
# Live world canvas
# ---------------------------------------------------------------------------
class SimCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.sim = None
        self.infection_radius = 9.0
        self.hospital_frac = 0.15
        self.infect_strain = 0
        self.show_map = False
        self.on_manual_infection = None
        self.setMinimumSize(300, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCursor(Qt.PointingHandCursor)

    def attach(self, sim):
        self.sim = sim

    def _transform(self):
        s = min(self.width() / WORLD_W, self.height() / WORLD_H)
        return s, (self.width() - WORLD_W * s) / 2, (self.height() - WORLD_H * s) / 2

    def mousePressEvent(self, e):
        if self.sim is None:
            return
        s, ox, oy = self._transform()
        if s <= 0:
            return
        wx, wy = (e.position().x() - ox) / s, (e.position().y() - oy) / s
        if 0 <= wx <= WORLD_W and 0 <= wy <= WORLD_H:
            hit_r = max(12.0, self.infection_radius * 1.5)
            if self.sim.infect_at(wx, wy, radius=hit_r,
                                  strain=self.infect_strain) and self.on_manual_infection:
                self.on_manual_infection()

    def _draw_halo(self, p, idx, r, color):
        path = QPainterPath()
        for i in idx:
            x, y = self.sim.pos[i]
            path.addEllipse(QPointF(float(x), float(y)), r, r)
        p.setPen(Qt.NoPen)
        p.setBrush(color)
        p.drawPath(path)

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0b0e13"))
        p.setRenderHint(QPainter.Antialiasing)
        s, ox, oy = self._transform()
        world = QRectF(ox, oy, WORLD_W * s, WORLD_H * s)
        p.fillRect(world, QColor("#0d1117"))
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor("#30363d"), 1))
        p.drawRect(world)
        if self.sim is None or self.sim.n_agents == 0:
            return

        p.translate(ox, oy)
        p.scale(s, s)
        sim = self.sim
        vis, vstr = sim.visual()
        r = float(self.infection_radius)
        map_colors = STRAIN_COLORS if sim.showdown else (STATE_COLORS[INFECTED],)

        if self.show_map:
            draw_conquest_map(p, sim, QRectF(0, 0, WORLD_W, WORLD_H), map_colors)

        alive = ~sim.dead
        ss = sim.strain_state
        for s_i in range(sim.n_strains):
            sick = np.flatnonzero(alive & (ss[s_i] == INFECTED))
            if sick.size == 0:
                continue
            sup = sim.super_spreader[sick]
            if (~sup).any():
                self._draw_halo(p, sick[~sup], r, _alpha(STRAIN_COLORS[s_i], 22))
            if sup.any():
                self._draw_halo(p, sick[sup], r * SUPER_SPREAD_FACTOR,
                                _alpha(STRAIN_COLORS[s_i], 36))
        sup_sus = np.flatnonzero((vis == SUSCEPTIBLE) & sim.super_spreader)
        if sup_sus.size:
            path = QPainterPath()
            for i in sup_sus:
                x, y = sim.pos[i]
                path.addEllipse(QPointF(float(x), float(y)), 6.5, 6.5)
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(227, 179, 65, 150), 1.2))
            p.drawPath(path)

        p.setPen(Qt.NoPen)
        for state in (RECOVERED, DEAD):
            idx = np.flatnonzero(vis == state)
            if idx.size:
                path = QPainterPath()
                rad = AGENT_RADIUS[state]
                for i in idx:
                    x, y = sim.pos[i]
                    path.addEllipse(QPointF(float(x), float(y)), rad, rad)
                p.setBrush(STATE_COLORS[state])
                p.drawPath(path)

        def draw_mask(mask, color, rad):
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                return
            path = QPainterPath()
            for i in idx:
                x, y = sim.pos[i]
                path.addEllipse(QPointF(float(x), float(y)), rad, rad)
            p.setBrush(color)
            p.drawPath(path)

        if sim.showdown:
            for s_i in range(sim.n_strains):
                draw_mask((vis == EXPOSED) & (vstr == s_i),
                          _dim(STRAIN_COLORS[s_i]), AGENT_RADIUS[EXPOSED])
        else:
            draw_mask(vis == EXPOSED, EXPOSED_COLOR, AGENT_RADIUS[EXPOSED])
        draw_mask(vis == SUSCEPTIBLE, STATE_COLORS[SUSCEPTIBLE],
                  AGENT_RADIUS[SUSCEPTIBLE])
        if sim.showdown:
            for s_i in range(sim.n_strains):
                draw_mask((vis == INFECTED) & (vstr == s_i),
                          STRAIN_COLORS[s_i], AGENT_RADIUS[INFECTED])
        else:
            draw_mask(vis == INFECTED, STATE_COLORS[INFECTED],
                      AGENT_RADIUS[INFECTED])

        # status badges (drawn in screen space)
        p.resetTransform()
        flags = []
        if sim.in_lockdown:
            flags.append((f"LOCKDOWN {sim.lockdown_remaining_days:.0f}d",
                          QColor("#58a6ff")))
        if sim.isolate_infected:
            flags.append(("ISOLATION ON", QColor("#3fb950")))
        pop = sim.n_agents
        if pop * self.hospital_frac > 0 and int((vis == INFECTED).sum()) > pop * self.hospital_frac:
            flags.append(("HOSPITALS OVERWHELMED — deaths ×4", QColor("#ff8181")))
        y = world.top() + 8
        p.setFont(QFont("", 9, QFont.Bold))
        fm = p.fontMetrics()
        for text, color in flags:
            w = fm.horizontalAdvance(text) + 20
            rect = QRectF(world.left() + 8, y, w, 22)
            p.setBrush(QColor(13, 17, 23, 225))
            p.setPen(QPen(color, 1.2))
            p.drawRoundedRect(rect, 11, 11)
            p.setPen(color)
            p.drawText(rect, Qt.AlignCenter, text)
            y += 26


# ---------------------------------------------------------------------------
# Live chart widget
# ---------------------------------------------------------------------------
class ChartWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._history = np.empty((0, 9), np.int64)
        self._population = 1
        self.series = []
        self.setFixedHeight(190)

    def set_series(self, spec):
        self.series = list(spec)
        self.update()

    def set_data(self, history, population):
        self._history = history
        self._population = max(int(population), 1)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0d1117"))
        w, h = self.width(), self.height()
        ml, mt, mr, mb = 46, 10, 14, 20
        pw, ph = w - ml - mr, h - mt - mb
        if pw < 20 or ph < 20 or not self.series:
            return
        p.setRenderHint(QPainter.Antialiasing)
        p.setFont(QFont("", 8))

        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor("#30363d"), 1))
        p.drawRect(QRectF(ml, mt, pw, ph))
        p.setPen(QPen(QColor("#1c2129"), 1))
        for frac in (0.25, 0.5, 0.75):
            y = round(mt + ph * (1 - frac))
            p.drawLine(ml, y, ml + pw, y)

        hist = self._history
        if hist.shape[0] >= 2:
            data = hist
            if data.shape[0] > 2400:
                stride = int(np.ceil(data.shape[0] / 2400))
                data = data[::stride]
            t = data[:, 0].astype(float)
            t_max = max(float(t[-1]), 1.0)
            n = self._population
            X = lambda tv: float(ml + pw * (tv / t_max))
            Y = lambda v: float(mt + ph * (1.0 - min(v / n, 1.0)))

            pen = QPen()
            pen.setWidthF(1.7)
            pen.setCapStyle(Qt.FlatCap)
            for col, color, _label in self.series:
                pen.setColor(color)
                p.setPen(pen)
                path = QPainterPath(QPointF(X(t[0]), Y(data[0, col])))
                for k in range(1, data.shape[0]):
                    path.lineTo(X(t[k]), Y(data[k, col]))
                p.drawPath(path)

        p.setPen(QColor("#8b949e"))
        p.drawText(QRectF(0, mt - 8, ml - 6, 16), Qt.AlignRight | Qt.AlignVCenter, f"{self._population:,}")
        p.drawText(QRectF(0, mt + ph / 2 - 8, ml - 6, 16), Qt.AlignRight | Qt.AlignVCenter, f"{self._population / 2:,.0f}")
        p.drawText(QRectF(0, mt + ph - 8, ml - 6, 16), Qt.AlignRight | Qt.AlignVCenter, "0")
        t_end = float(self._history[-1, 0]) if self._history.shape[0] else 0.0
        p.drawText(QRectF(ml, mt + ph + 2, 120, 16), Qt.AlignLeft | Qt.AlignVCenter, "Day 0")
        p.drawText(QRectF(ml + pw - 120, mt + ph + 2, 120, 16), Qt.AlignRight | Qt.AlignVCenter,
                   f"Day {t_end / TICKS_PER_DAY:.0f}")

        x = ml + 10
        for _col, color, label in self.series:
            p.fillRect(QRectF(x, mt + 8, 9, 9), color)
            adv = p.fontMetrics().horizontalAdvance(label)
            p.drawText(QRectF(x + 13, mt + 3, adv + 8, 16), Qt.AlignLeft | Qt.AlignVCenter, label)
            x += 13 + adv + 14


# ---------------------------------------------------------------------------
# Shorts frame renderer (vertical 9:16, painted offscreen)
# ---------------------------------------------------------------------------
class ShortsRenderer:
    def __init__(self, width, height, title, meta):
        self.w, self.h = int(width), int(height)
        self.k = self.w / 1080.0
        self.title = title
        self.meta = meta
        self.img = QImage(self.w, self.h, QImage.Format_RGB888)

    def render(self, sim, epidemic_over, frame_idx=0, max_frames=0,
               rt_text=None, fade_frames=0):
        k = self.k
        img = self.img
        img.fill(QColor("#0b0e13"))
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)

        m = 40 * k
        fade = min(1.0, frame_idx / fade_frames) if fade_frames > 0 else 1.0
        self._draw_header(p, fade)
        world = QRectF(m, 245 * k, self.w - 2 * m, 600 * k)
        self._draw_world(p, sim, world)
        self._draw_stats(p, sim, world.bottom() + 45 * k)
        chart = QRectF(m, 1055 * k, self.w - 2 * m, 645 * k)
        self._draw_chart(p, sim, chart)
        self._draw_footer(p, rt_text)
        if epidemic_over:
            self._draw_endcard(p, sim, world)
        if max_frames > 0:
            frac = min(1.0, frame_idx / max_frames)
            if frac > 0:
                p.fillRect(QRectF(0, self.h - 6 * k, self.w * frac, 6 * k),
                           QColor("#1f6feb"))

        p.end()
        return qimage_to_array(img)

    def _draw_header(self, p, fade=1.0):
        k = self.k
        p.setOpacity(fade)
        p.setPen(QColor("#f0f6fc"))
        p.setFont(_px_font(64 * k, True))
        fm = p.fontMetrics()
        title = fm.elidedText(self.title, Qt.ElideRight, int(self.w - 60 * k))
        p.drawText(QRectF(0, 62 * k, self.w, 84 * k), Qt.AlignHCenter | Qt.AlignVCenter, title)
        label = self.meta.get("label")
        if label:
            p.setPen(QColor("#e3b341"))
            p.setFont(_px_font(31 * k, True))
            fm = p.fontMetrics()
            label = fm.elidedText(label, Qt.ElideRight, int(self.w - 60 * k))
            p.drawText(QRectF(0, 152 * k, self.w, 42 * k), Qt.AlignHCenter | Qt.AlignVCenter, label)
        else:
            sub = "multi-virus showdown" if self.meta["showdown"] else "agent-based epidemic simulation"
            p.setPen(QColor("#8b949e"))
            p.setFont(_px_font(27 * k))
            p.drawText(QRectF(0, 152 * k, self.w, 36 * k), Qt.AlignHCenter | Qt.AlignVCenter, sub)
        p.setOpacity(1.0)

    def _draw_world(self, p, sim, rect):
        k = self.k
        p.fillRect(rect, QColor("#0d1117"))
        p.setPen(QPen(QColor("#30363d"), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(rect)

        s = rect.width() / WORLD_W
        p.save()
        p.translate(rect.left(), rect.top())
        p.scale(s, s)
        vis, vstr = sim.visual()
        r_base = float(self.meta["radius"])
        alive = ~sim.dead
        ss = sim.strain_state

        if self.meta.get("show_map"):
            draw_conquest_map(p, sim, QRectF(0, 0, WORLD_W, WORLD_H),
                              STRAIN_COLORS if self.meta["showdown"]
                              else (STATE_COLORS[INFECTED],))

        def halo(mask, radius, color):
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                return
            path = QPainterPath()
            for i in idx:
                x, y = sim.pos[i]
                path.addEllipse(QPointF(float(x), float(y)), radius, radius)
            p.setPen(Qt.NoPen)
            p.setBrush(color)
            p.drawPath(path)

        for s_i in range(sim.n_strains):
            sick = alive & (ss[s_i] == INFECTED)
            sup = sick & sim.super_spreader
            halo(sick & ~sup, r_base, _alpha(STRAIN_COLORS[s_i], 22))
            halo(sup, r_base * SUPER_SPREAD_FACTOR, _alpha(STRAIN_COLORS[s_i], 36))

        rings = np.flatnonzero((vis == SUSCEPTIBLE) & sim.super_spreader)
        if rings.size:
            path = QPainterPath()
            for i in rings:
                x, y = sim.pos[i]
                path.addEllipse(QPointF(float(x), float(y)), 6.5, 6.5)
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(227, 179, 65, 150), 1.2))
            p.drawPath(path)

        def dots(mask, color, rad):
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                return
            path = QPainterPath()
            for i in idx:
                x, y = sim.pos[i]
                path.addEllipse(QPointF(float(x), float(y)), rad, rad)
            p.setPen(Qt.NoPen)
            p.setBrush(color)
            p.drawPath(path)

        dots(vis == RECOVERED, STATE_COLORS[RECOVERED], AGENT_RADIUS[RECOVERED])
        dots(vis == DEAD, STATE_COLORS[DEAD], AGENT_RADIUS[DEAD])
        if self.meta["showdown"]:
            for s_i in range(sim.n_strains):
                dots((vis == EXPOSED) & (vstr == s_i),
                     _dim(STRAIN_COLORS[s_i]), AGENT_RADIUS[EXPOSED])
        else:
            dots(vis == EXPOSED, EXPOSED_COLOR, AGENT_RADIUS[EXPOSED])
        dots(vis == SUSCEPTIBLE, STATE_COLORS[SUSCEPTIBLE], AGENT_RADIUS[SUSCEPTIBLE])
        if self.meta["showdown"]:
            for s_i in range(sim.n_strains):
                dots((vis == INFECTED) & (vstr == s_i),
                     STRAIN_COLORS[s_i], AGENT_RADIUS[INFECTED])
        else:
            dots(vis == INFECTED, STATE_COLORS[INFECTED], AGENT_RADIUS[INFECTED])
        p.restore()

        day = sim.step_count / TICKS_PER_DAY
        self._chip(p, QRectF(rect.left() + 14 * k, rect.top() + 14 * k, 150 * k, 46 * k),
                   f"DAY {day:.1f}", QColor("#e3b341"))
        S, E, I, R, D, sick = sim.count_blocks()
        x_right = rect.right() - 14 * k
        p.setFont(_px_font(21 * k, True))
        fm = p.fontMetrics()
        if self.meta["showdown"]:
            for s_i in reversed(range(self.meta["n_strains"])):
                text = f"{STRAIN_NAMES[s_i]} {sick[s_i]:,}"
                cw = fm.horizontalAdvance(text) + 30 * k
                self._chip(p, QRectF(x_right - cw, rect.top() + 14 * k, cw, 46 * k),
                           text, STRAIN_COLORS[s_i])
                x_right -= cw + 10 * k
        else:
            text = f"INFECTED {I + E:,}"
            cw = fm.horizontalAdvance(text) + 34 * k
            self._chip(p, QRectF(x_right - cw, rect.top() + 14 * k, cw, 46 * k),
                       text, QColor("#ff5c5c"))

        if self.meta.get("show_map"):
            self._chip(p, QRectF(rect.left() + 14 * k, rect.bottom() - 58 * k,
                                 210 * k, 44 * k),
                       "TERRITORY MAP", QColor("#8b949e"))

        # stacked alert chips, centered
        pop = max(sim.n_agents, 1)
        cap = pop * self.meta.get("hospital_frac", 0.0)
        alerts = []
        if cap > 0 and I > cap:
            alerts.append(("⚠ HOSPITALS OVERWHELMED", QColor("#ff5c5c")))
        if sim.in_lockdown:
            alerts.append((f"LOCKDOWN {sim.lockdown_remaining_days:.0f}d",
                           QColor("#58a6ff")))
        if self.meta.get("isolate") or sim.isolate_infected:
            alerts.append(("ISOLATION", QColor("#3fb950")))
        y = rect.top() + 68 * k
        for text, color in alerts:
            p.setFont(_px_font(21 * k, True))
            fm = p.fontMetrics()
            cw = fm.horizontalAdvance(text) + 34 * k
            self._chip(p, QRectF(rect.center().x() - cw / 2, y, cw, 46 * k),
                       text, color)
            y += 54 * k

    def _chip(self, p, rect, text, accent):
        k = self.k
        p.setBrush(QColor(13, 17, 23, 215))
        p.setPen(QPen(accent, 1.4))
        p.drawRoundedRect(rect, rect.height() / 2, rect.height() / 2)
        p.setFont(_px_font(21 * k, True))
        p.setPen(QColor("#e6edf3"))
        p.drawText(rect, Qt.AlignCenter, text)

    def _draw_stats(self, p, sim, y):
        k = self.k
        m = 40 * k
        n = max(sim.n_agents, 1)
        S, E, I, R, D, sick = sim.count_blocks()
        if self.meta["showdown"]:
            blocks = [(S, STATE_COLORS[SUSCEPTIBLE], "SUSCEPTIBLE")]
            for i in range(self.meta["n_strains"]):
                blocks.append((sick[i], STRAIN_COLORS[i], STRAIN_NAMES[i].upper()))
            blocks += [(R, STATE_COLORS[RECOVERED], "RECOVERED"),
                       (D, STATE_COLORS[DEAD], "DEAD")]
        else:
            blocks = [(S, STATE_COLORS[SUSCEPTIBLE], "SUSCEPTIBLE"),
                      (E, EXPOSED_COLOR, "EXPOSED"),
                      (I, STATE_COLORS[INFECTED], "INFECTED"),
                      (R, STATE_COLORS[RECOVERED], "RECOVERED"),
                      (D, STATE_COLORS[DEAD], "DEAD")]

        bar = QRectF(m, y, self.w - 2 * m, 42 * k)
        clip = QPainterPath()
        clip.addRoundedRect(bar, 21 * k, 21 * k)
        p.save()
        p.setClipPath(clip)
        x = bar.left()
        for cnt, color, _lbl in blocks:
            seg = bar.width() * cnt / n
            if seg > 0:
                p.fillRect(QRectF(x, bar.top(), seg + 1.0, bar.height()), color)
            x += seg
        p.restore()
        p.setPen(QPen(QColor("#30363d"), 1))
        p.setBrush(Qt.NoBrush)
        p.drawPath(clip)

        p.setFont(_px_font(20 * k, True))
        x = bar.left()
        for cnt, color, _lbl in blocks:
            seg = bar.width() * cnt / n
            if seg > 62 * k:
                p.setPen(QColor("#0d1117"))
                p.drawText(QRectF(x, bar.top(), seg, bar.height()),
                           Qt.AlignCenter, f"{100 * cnt / n:.0f}%")
            x += seg

        colw = (self.w - 2 * m) / len(blocks)
        for j, (cnt, color, label) in enumerate(blocks):
            cx = m + colw * j
            p.setFont(_px_font(40 * k, True))
            p.setPen(color)
            p.drawText(QRectF(cx, bar.bottom() + 14 * k, colw, 50 * k),
                       Qt.AlignHCenter | Qt.AlignVCenter, f"{cnt:,}")
            p.setFont(_px_font(19 * k))
            p.setPen(QColor("#6e7681"))
            p.drawText(QRectF(cx, bar.bottom() + 62 * k, colw, 26 * k),
                       Qt.AlignHCenter | Qt.AlignVCenter, label)

    def _chart_series(self):
        if self.meta["showdown"]:
            series = [(1, STATE_COLORS[SUSCEPTIBLE], "Susceptible")]
            for i in range(self.meta["n_strains"]):
                series.append((6 + i, STRAIN_COLORS[i], STRAIN_NAMES[i]))
            series += [(4, STATE_COLORS[RECOVERED], "Recovered"),
                       (5, STATE_COLORS[DEAD], "Dead")]
        else:
            series = [(1, STATE_COLORS[SUSCEPTIBLE], "Susceptible"),
                      (2, EXPOSED_COLOR, "Exposed"),
                      (3, STATE_COLORS[INFECTED], "Infected"),
                      (4, STATE_COLORS[RECOVERED], "Recovered"),
                      (5, STATE_COLORS[DEAD], "Dead")]
        return series

    def _draw_chart(self, p, sim, rect):
        k = self.k
        p.fillRect(rect, QColor("#0d1117"))
        p.setPen(QPen(QColor("#30363d"), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(rect)
        pop = max(sim.n_agents, 1)
        pad_l, pad_t, pad_b, pad_r = 70 * k, 18 * k, 34 * k, 12 * k
        plot = QRectF(rect.left() + pad_l, rect.top() + pad_t,
                      rect.width() - pad_l - pad_r, rect.height() - pad_t - pad_b)

        p.setFont(_px_font(20 * k))
        for frac in (1.0, 0.5, 0.0):
            yv = plot.top() + plot.height() * (1 - frac)
            label = f"{pop:,}" if frac == 1.0 else (f"{pop // 2:,}" if frac == 0.5 else "0")
            p.setPen(QColor("#8b949e"))
            p.drawText(QRectF(rect.left() + 4 * k, yv - 10 * k, pad_l - 10 * k, 20 * k),
                       Qt.AlignRight | Qt.AlignVCenter, label)
            if 0.0 < frac < 1.0:
                p.setPen(QPen(QColor("#1c2129"), 1))
                p.drawLine(QPointF(plot.left(), yv), QPointF(plot.right(), yv))

        hist = sim.history
        series = self._chart_series()
        if hist.shape[0] >= 2:
            data = hist
            if data.shape[0] > 1500:
                stride = int(np.ceil(data.shape[0] / 1500))
                data = data[::stride]
            t = data[:, 0].astype(float)
            t_max = max(float(t[-1]), 1.0)
            pen = QPen()
            pen.setWidthF(max(1.6, 2.2 * k))
            pen.setCapStyle(Qt.FlatCap)
            for col, color, _lbl in series:
                pen.setColor(color)
                p.setPen(pen)
                x0 = plot.left()
                y0 = plot.bottom() - plot.height() * min(data[0, col] / pop, 1.0)
                path = QPainterPath(QPointF(x0, y0))
                for row in range(1, data.shape[0]):
                    x = plot.left() + plot.width() * t[row] / t_max
                    y = plot.bottom() - plot.height() * min(data[row, col] / pop, 1.0)
                    path.lineTo(x, y)
                p.drawPath(path)

        t_end = float(hist[-1, 0]) if hist.shape[0] else 0.0
        p.setFont(_px_font(20 * k))
        p.setPen(QColor("#8b949e"))
        p.drawText(QRectF(plot.left(), rect.bottom() - pad_b + 2 * k, 160 * k, 24 * k),
                   Qt.AlignLeft | Qt.AlignVCenter, "Day 0")
        p.drawText(QRectF(plot.right() - 200 * k, rect.bottom() - pad_b + 2 * k, 200 * k, 24 * k),
                   Qt.AlignRight | Qt.AlignVCenter, f"Day {t_end / TICKS_PER_DAY:.0f}")

        p.setFont(_px_font(19 * k))
        fm = p.fontMetrics()
        lx = plot.left() + 6 * k
        for _col, color, label in series:
            p.setPen(Qt.NoPen)
            p.setBrush(color)
            p.drawRoundedRect(QRectF(lx, rect.top() + 7 * k, 14 * k, 14 * k), 3 * k, 3 * k)
            p.setPen(QColor("#9aa4ae"))
            adv = fm.horizontalAdvance(label)
            p.drawText(QRectF(lx + 20 * k, rect.top(), adv + 8 * k, 28 * k),
                       Qt.AlignLeft | Qt.AlignVCenter, label)
            lx += 20 * k + adv + 16 * k

    def _draw_footer(self, p, rt_text=None):
        k = self.k
        meta = self.meta
        p.setFont(_px_font(21 * k))
        p.setPen(QColor("#6e7681"))

        def fmt_strain(sp):
            rec = sp["recovery_days"]
            rec_txt = "never" if rec == 0 else f"{rec} d"
            return (f"{sp['name']}: transmit {sp['chance'] * 100:.4g}% · "
                    f"incubation {sp['incubation_days']} d · recovery {rec_txt} · "
                    f"mortality {sp['mortality'] * 100:.4g}%")

        if meta["showdown"]:
            y = 1704 * k
            lines = [fmt_strain(sp) for sp in meta["strains"]]
            ci = meta.get("cross_immunity", 0.0)
            if ci > 0:
                lines.append(f"cross-immunity {ci * 100:.0f}%")
            if rt_text:
                lines.append(rt_text)
            for line in lines[:6]:
                p.drawText(QRectF(0, y, self.w, 30 * k),
                           Qt.AlignHCenter | Qt.AlignVCenter, line)
                y += 33 * k
        else:
            sp = meta["strains"][0]
            text = f"population {meta['population']:,} · " + fmt_strain(sp)
            if rt_text:
                text += f" · {rt_text}"
            p.drawText(QRectF(0, 1790 * k, self.w, 34 * k),
                       Qt.AlignHCenter | Qt.AlignVCenter, text)

    def _draw_endcard(self, p, sim, world):
        k = self.k
        p.fillRect(world, QColor(5, 8, 12, 200))
        row = sim.history[-1]
        t, r, d = int(row[0]), int(row[4]), int(row[5])
        pop = max(sim.n_agents, 1)
        p.setPen(QColor("#ff5c5c"))
        p.setFont(_px_font(64 * k, True))
        p.drawText(QRectF(world.left(), world.top() + 100 * k, world.width(), 90 * k),
                   Qt.AlignHCenter, "EPIDEMIC OVER")

        if self.meta["showdown"] and sim.n_strains > 1:
            ever = sim.ever_per_strain
            total = int(ever.sum())
            if total > 0:
                w = int(np.argmax(ever))
                share = float(ever[w]) / total
                p.setPen(STRAIN_COLORS[w])
                p.setFont(_px_font(62 * k, True))
                p.drawText(QRectF(world.left(), world.top() + 205 * k, world.width(), 84 * k),
                           Qt.AlignHCenter, f"{STRAIN_NAMES[w].upper()} WINS")
                p.setPen(QColor("#f0f6fc"))
                p.setFont(_px_font(36 * k))
                p.drawText(QRectF(world.left(), world.top() + 295 * k, world.width(), 56 * k),
                           Qt.AlignHCenter,
                           f"{share * 100:.0f}% of all infections ({int(ever[w]):,})")
            else:
                p.setPen(QColor("#f0f6fc"))
                p.setFont(_px_font(40 * k, True))
                p.drawText(QRectF(world.left(), world.top() + 220 * k, world.width(), 60 * k),
                           Qt.AlignHCenter, "CONTAINED")
            y = 365 * k
        else:
            p.setPen(QColor("#f0f6fc"))
            p.setFont(_px_font(38 * k))
            p.drawText(QRectF(world.left(), world.top() + 220 * k, world.width(), 60 * k),
                       Qt.AlignHCenter,
                       f"{100 * sim.ever_infected / pop:.0f}% of the population was infected")
            y = 300 * k

        p.setPen(STATE_COLORS[RECOVERED])
        p.setFont(_px_font(34 * k))
        p.drawText(QRectF(world.left(), world.top() + y, world.width(), 54 * k),
                   Qt.AlignHCenter, f"{r:,} recovered")
        p.setPen(STATE_COLORS[DEAD])
        p.drawText(QRectF(world.left(), world.top() + y + 58 * k, world.width(), 54 * k),
                   Qt.AlignHCenter, f"{d:,} died")
        p.setPen(QColor("#8b949e"))
        p.setFont(_px_font(27 * k))
        p.drawText(QRectF(world.left(), world.top() + y + 120 * k, world.width(), 44 * k),
                   Qt.AlignHCenter,
                   f"peak {sim.peak_infected:,} sick at once · day {t / TICKS_PER_DAY:.0f}")


# ---------------------------------------------------------------------------
# Head-to-head split-screen renderer
# ---------------------------------------------------------------------------
class SplitRenderer:
    def __init__(self, width, height, title, criterion_text, sides, show_map):
        self.w, self.h = int(width), int(height)
        self.k = self.w / 1080.0
        self.title = title
        self.criterion_text = criterion_text
        self.sides = sides
        self.show_map = show_map
        self.img = QImage(self.w, self.h, QImage.Format_RGB888)

    def render(self, simL, simR, overL, overR, verdict=None,
               frame_idx=0, max_frames=0, rt_text=None, fade_frames=0):
        k = self.k
        img = self.img
        img.fill(QColor("#0b0e13"))
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)

        m = 40 * k
        wW = (self.w - 3 * m) / 2
        wH = wW * WORLD_H / WORLD_W
        world_y = 246 * k
        rects = (QRectF(m, world_y, wW, wH),
                 QRectF(m * 2 + wW, world_y, wW, wH))
        sims = (simL, simR)
        overs = (overL, overR)

        fade = min(1.0, frame_idx / fade_frames) if fade_frames > 0 else 1.0
        self._draw_header(p, fade)
        for i in range(2):
            self._draw_side_chip(p, i, rects[i])
            self._draw_world(p, sims[i], rects[i], i, overs[i])
        self._draw_vs(p, rects[0])
        self._draw_statrows(p, sims)
        chart = QRectF(m, 786 * k, self.w - 2 * m, 742 * k)
        self._draw_chart(p, sims, chart)
        self._draw_footer(p, rt_text)
        if verdict is not None:
            self._draw_endcard(p, verdict, world_y - 20 * k)
        if max_frames > 0:
            frac = min(1.0, frame_idx / max_frames)
            if frac > 0:
                p.fillRect(QRectF(0, self.h - 6 * k, self.w * frac, 6 * k),
                           QColor("#1f6feb"))

        p.end()
        return qimage_to_array(img)

    def _draw_header(self, p, fade=1.0):
        k = self.k
        p.setOpacity(fade)
        p.setPen(QColor("#f0f6fc"))
        p.setFont(_px_font(56 * k, True))
        fm = p.fontMetrics()
        title = fm.elidedText(self.title, Qt.ElideRight, int(self.w - 60 * k))
        p.drawText(QRectF(0, 52 * k, self.w, 82 * k), Qt.AlignHCenter | Qt.AlignVCenter, title)
        p.setPen(QColor("#e3b341"))
        p.setFont(_px_font(26 * k, True))
        p.drawText(QRectF(0, 140 * k, self.w, 36 * k), Qt.AlignHCenter | Qt.AlignVCenter,
                   f"HEAD-TO-HEAD · {self.criterion_text}")
        p.setOpacity(1.0)

    def _draw_side_chip(self, p, i, world_rect):
        k = self.k
        accent = self.sides[i]["color"]
        text = self.sides[i]["label"]
        p.setFont(_px_font(26 * k, True))
        fm = p.fontMetrics()
        text = fm.elidedText(text, Qt.ElideRight, int(world_rect.width() - 30 * k))
        cw = min(world_rect.width(), fm.horizontalAdvance(text) + 36 * k)
        rect = QRectF(world_rect.center().x() - cw / 2, 186 * k, cw, 44 * k)
        p.setBrush(QColor(13, 17, 23, 225))
        p.setPen(QPen(accent, 1.6))
        p.drawRoundedRect(rect, rect.height() / 2, rect.height() / 2)
        p.setPen(QColor("#e6edf3"))
        p.drawText(rect, Qt.AlignCenter, text)

    def _draw_world(self, p, sim, rect, i, over):
        k = self.k
        accent = self.sides[i]["color"]
        p.fillRect(rect, QColor("#0d1117"))
        p.setPen(QPen(accent, 2))
        p.setBrush(Qt.NoBrush)
        p.drawRect(rect)

        s = rect.width() / WORLD_W
        p.save()
        p.translate(rect.left(), rect.top())
        p.scale(s, s)
        if self.show_map:
            draw_conquest_map(p, sim, QRectF(0, 0, WORLD_W, WORLD_H),
                              (STATE_COLORS[INFECTED],))
        r_base = float(self.sides[i]["radius"])
        alive = ~sim.dead
        sick = np.flatnonzero(alive & (sim.strain_state[0] == INFECTED))
        if sick.size:
            sup = sim.super_spreader[sick]

            def halo(idx, r, color):
                if idx.size == 0:
                    return
                path = QPainterPath()
                for j in idx:
                    x, y = sim.pos[j]
                    path.addEllipse(QPointF(float(x), float(y)), r, r)
                p.setPen(Qt.NoPen)
                p.setBrush(color)
                p.drawPath(path)

            halo(sick[~sup], r_base, QColor(255, 92, 92, 24))
            halo(sick[sup], r_base * SUPER_SPREAD_FACTOR, QColor(255, 92, 92, 40))
        vis, _vstr = sim.visual()

        def dots(mask, color, rad):
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                return
            path = QPainterPath()
            for j in idx:
                x, y = sim.pos[j]
                path.addEllipse(QPointF(float(x), float(y)), rad, rad)
            p.setPen(Qt.NoPen)
            p.setBrush(color)
            p.drawPath(path)

        dots(vis == RECOVERED, STATE_COLORS[RECOVERED], AGENT_RADIUS[RECOVERED])
        dots(vis == DEAD, STATE_COLORS[DEAD], AGENT_RADIUS[DEAD])
        dots(vis == EXPOSED, _dim(EXPOSED_COLOR), AGENT_RADIUS[EXPOSED])
        dots(vis == SUSCEPTIBLE, STATE_COLORS[SUSCEPTIBLE], AGENT_RADIUS[SUSCEPTIBLE])
        dots(vis == INFECTED, STATE_COLORS[INFECTED], AGENT_RADIUS[INFECTED])
        p.restore()

        day = sim.step_count / TICKS_PER_DAY
        self._mini_chip(p, QRectF(rect.left() + 10 * k, rect.top() + 10 * k,
                                  104 * k, 38 * k), f"D {day:.0f}", QColor("#e3b341"))
        n_sick = int((vis == INFECTED).sum()) + int((vis == EXPOSED).sum())
        p.setFont(_px_font(18 * k, True))
        fm = p.fontMetrics()
        text = f"SICK {n_sick:,}"
        cw = fm.horizontalAdvance(text) + 26 * k
        self._mini_chip(p, QRectF(rect.right() - 10 * k - cw, rect.top() + 10 * k, cw, 38 * k),
                        text, accent)
        if over:
            p.setFont(_px_font(20 * k, True))
            fm = p.fontMetrics()
            text = "OUTBREAK OVER"
            cw = fm.horizontalAdvance(text) + 30 * k
            self._mini_chip(p, QRectF(rect.center().x() - cw / 2,
                                      rect.center().y() - 21 * k, cw, 42 * k),
                            text, QColor("#3fb950"))

    def _mini_chip(self, p, rect, text, accent):
        k = self.k
        p.setBrush(QColor(13, 17, 23, 220))
        p.setPen(QPen(accent, 1.2))
        p.drawRoundedRect(rect, rect.height() / 2, rect.height() / 2)
        p.setFont(_px_font(18 * k, True))
        p.setPen(QColor("#e6edf3"))
        p.drawText(rect, Qt.AlignCenter, text)

    def _draw_vs(self, p, world_rect):
        k = self.k
        cx = self.w / 2
        cy = world_rect.center().y()
        r = 34 * k
        p.setBrush(QColor(13, 17, 23, 240))
        p.setPen(QPen(QColor("#e6edf3"), 2))
        p.drawEllipse(QPointF(cx, cy), r, r)
        p.setFont(_px_font(28 * k, True))
        p.setPen(QColor("#f0f6fc"))
        p.drawText(QRectF(cx - r, cy - r, 2 * r, 2 * r), Qt.AlignCenter, "VS")

    def _draw_statrows(self, p, sims):
        k = self.k
        m = 40 * k
        names = ("SICK", "RECOVERED", "DEAD", "DAY")
        colw = (self.w - 2 * m - 26 * k) / 4
        for i, sim in enumerate(sims):
            top = (556 + i * 108) * k
            accent = self.sides[i]["color"]
            p.fillRect(QRectF(m, top, 6 * k, 96 * k), accent)
            vis, _ = sim.visual()
            E = int((vis == EXPOSED).sum()); I = int((vis == INFECTED).sum())
            R = int((vis == RECOVERED).sum()); D = int((vis == DEAD).sum())
            row = sim.history[-1]
            vals = (E + I, R, D, f"{row[0] / TICKS_PER_DAY:.0f}")
            colors = (accent, STATE_COLORS[RECOVERED], STATE_COLORS[DEAD],
                      QColor("#e3b341"))
            for j, (val, color, name) in enumerate(zip(vals, colors, names)):
                x = m + 26 * k + j * colw
                p.setFont(_px_font(36 * k, True))
                p.setPen(color)
                p.drawText(QRectF(x, top, colw, 52 * k), Qt.AlignLeft | Qt.AlignVCenter,
                           f"{val:,}" if isinstance(val, int) else val)
                p.setFont(_px_font(16 * k))
                p.setPen(QColor("#6e7681"))
                p.drawText(QRectF(x, top + 52 * k, colw, 24 * k),
                           Qt.AlignLeft | Qt.AlignVCenter, name)

    def _draw_chart(self, p, sims, rect):
        k = self.k
        p.fillRect(rect, QColor("#0d1117"))
        p.setPen(QPen(QColor("#30363d"), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(rect)
        pop = max(sims[0].n_agents, 1)
        pad_l, pad_t, pad_b, pad_r = 74 * k, 46 * k, 34 * k, 14 * k
        plot = QRectF(rect.left() + pad_l, rect.top() + pad_t,
                      rect.width() - pad_l - pad_r, rect.height() - pad_t - pad_b)

        p.setFont(_px_font(19 * k))
        for frac in (1.0, 0.5, 0.0):
            yv = plot.top() + plot.height() * (1 - frac)
            label = f"{pop:,}" if frac == 1.0 else (f"{pop // 2:,}" if frac == 0.5 else "0")
            p.setPen(QColor("#8b949e"))
            p.drawText(QRectF(rect.left() + 6 * k, yv - 11 * k, pad_l - 12 * k, 22 * k),
                       Qt.AlignRight | Qt.AlignVCenter, label)
            if 0.0 < frac < 1.0:
                p.setPen(QPen(QColor("#1c2129"), 1))
                p.drawLine(QPointF(plot.left(), yv), QPointF(plot.right(), yv))

        p.setFont(_px_font(19 * k, True))
        p.setPen(QColor("#6e7681"))
        p.drawText(QRectF(rect.left() + 10 * k, rect.top() + 8 * k, plot.width(), 24 * k),
                   Qt.AlignLeft | Qt.AlignVCenter, "ACTIVE INFECTIONS")

        t_max = max(float(sims[0].history[-1, 0]), float(sims[1].history[-1, 0]), 1.0)
        pen = QPen()
        pen.setWidthF(max(1.8, 2.6 * k))
        pen.setCapStyle(Qt.FlatCap)
        for i, sim in enumerate(sims):
            hist = sim.history
            if hist.shape[0] < 2:
                continue
            data = hist
            if data.shape[0] > 1500:
                stride = int(np.ceil(data.shape[0] / 1500))
                data = data[::stride]
            t = data[:, 0].astype(float)
            active = data[:, 2] + data[:, 3]
            pen.setColor(self.sides[i]["color"])
            p.setPen(pen)
            y0 = plot.bottom() - plot.height() * min(active[0] / pop, 1.0)
            path = QPainterPath(QPointF(plot.left(), y0))
            for r in range(1, data.shape[0]):
                x = plot.left() + plot.width() * t[r] / t_max
                y = plot.bottom() - plot.height() * min(active[r] / pop, 1.0)
                path.lineTo(x, y)
            p.drawPath(path)

        p.setFont(_px_font(19 * k))
        p.setPen(QColor("#8b949e"))
        p.drawText(QRectF(plot.left(), rect.bottom() - pad_b + 2 * k, 150 * k, 22 * k),
                   Qt.AlignLeft | Qt.AlignVCenter, "Day 0")
        p.drawText(QRectF(plot.right() - 190 * k, rect.bottom() - pad_b + 2 * k, 190 * k, 22 * k),
                   Qt.AlignRight | Qt.AlignVCenter, f"Day {t_max / TICKS_PER_DAY:.0f}")

        p.setFont(_px_font(19 * k, True))
        fm = p.fontMetrics()
        lx = plot.left() + 8 * k
        ly = plot.top() + 6 * k
        for i in range(2):
            label = fm.elidedText(self.sides[i]["label"], Qt.ElideRight,
                                  int(plot.width() / 2 - 60 * k))
            p.setPen(Qt.NoPen)
            p.setBrush(self.sides[i]["color"])
            p.drawRoundedRect(QRectF(lx, ly, 16 * k, 16 * k), 4 * k, 4 * k)
            p.setPen(QColor("#c9d1d9"))
            adv = fm.horizontalAdvance(label)
            p.drawText(QRectF(lx + 22 * k, ly - 4 * k, adv + 10 * k, 24 * k),
                       Qt.AlignLeft | Qt.AlignVCenter, label)
            lx += 22 * k + adv + 30 * k

    def _draw_footer(self, p, rt_text=None):
        k = self.k
        for i in range(2):
            y = (1548 + i * 36) * k
            p.setFont(_px_font(21 * k))
            p.setPen(QColor("#6e7681"))
            fm = p.fontMetrics()
            text = f"{self.sides[i]['label']} — {self.sides[i]['paramline']}"
            if i == 0 and rt_text:
                text += f" · {rt_text}"
            text = fm.elidedText(text, Qt.ElideRight, int(self.w - 40 * k))
            p.drawText(QRectF(0, y, self.w, 30 * k), Qt.AlignHCenter | Qt.AlignVCenter, text)

    def _draw_endcard(self, p, verdict, top):
        k = self.k
        p.fillRect(QRectF(0, top, self.w, 1302 * k), QColor(5, 8, 12, 216))
        stats = verdict["stats"]
        win = verdict["winner"]
        if win < 0:
            p.setPen(QColor("#f0f6fc"))
            p.setFont(_px_font(64 * k, True))
            p.drawText(QRectF(0, top + 140 * k, self.w, 90 * k),
                       Qt.AlignHCenter, "IT'S A DRAW")
        else:
            p.setPen(self.sides[win]["color"])
            p.setFont(_px_font(66 * k, True))
            fm = p.fontMetrics()
            label = fm.elidedText(self.sides[win]["label"].upper(),
                                  Qt.ElideRight, int(self.w - 60 * k))
            p.drawText(QRectF(0, top + 140 * k, self.w, 94 * k),
                       Qt.AlignHCenter, f"{label} WINS")
        p.setPen(QColor("#8b949e"))
        p.setFont(_px_font(27 * k))
        p.drawText(QRectF(0, top + 248 * k, self.w, 38 * k),
                   Qt.AlignHCenter, verdict["criterion"])
        for i in range(2):
            st = stats[i]
            color = self.sides[i]["color"] if win == i else QColor("#c9d1d9")
            p.setPen(color)
            p.setFont(_px_font(28 * k, True))
            fm = p.fontMetrics()
            text = fm.elidedText(
                f"{self.sides[i]['label']}:  {st['infected']:,} infected "
                f"({100 * st['infected'] / max(st['pop'], 1):.0f}%) · "
                f"{st['deaths']:,} dead · peak {st['peak']:,} · day {st['days']:.0f}",
                Qt.ElideRight, int(self.w - 50 * k))
            p.drawText(QRectF(0, top + (320 + i * 52) * k, self.w, 44 * k),
                       Qt.AlignHCenter, text)


# ---------------------------------------------------------------------------
# Batch render thread
# ---------------------------------------------------------------------------
class BatchRenderThread(QThread):
    progress = Signal(int, int, int, int)
    completed = Signal(object, str)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, configs, seed=None, parent=None):
        super().__init__(parent)
        self.configs = configs
        self.seed = seed
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            import imageio.v2 as imageio
        except ImportError:
            try:
                import imageio
            except ImportError:
                self.failed.emit(
                    "Video export needs imageio + its ffmpeg backend:\n"
                    "    pip install imageio imageio-ffmpeg")
                return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = []
        total = len(self.configs)
        for idx, cfg in enumerate(self.configs):
            if self._cancel:
                break
            try:
                path = self._render_one(imageio, cfg, stamp, idx, total)
            except Exception as exc:
                self.failed.emit(str(exc))
                return
            if path is None:
                break
            results.append((cfg.get("label") or f"variant {idx + 1}", path))
        if self._cancel or not results:
            self.cancelled.emit()
            return
        manifest = self._write_manifest(results, stamp)
        self.completed.emit(results, manifest)

    def _write_manifest(self, results, stamp):
        try:
            folder = os.path.dirname(os.path.abspath(results[0][1]))
            path = os.path.join(folder, f"virus_shorts_batch_{stamp}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"YouTube Shorts batch — {stamp}\n")
                f.write(f"Base title: {self.configs[0].get('title', '')}\n")
                f.write(f"Seed: {self.seed if self.seed is not None else 'random per variant'}\n\n")
                for label, file_path in results:
                    f.write(f"{label}\t{file_path}\n")
            return path
        except OSError:
            return ""

    def _render_one(self, imageio, s, stamp, idx, total):
        fps = int(s["fps"])
        max_frames = max(30, int(s["max_seconds"] * fps))
        end_hold = int(fps * 2.5)
        tpf = max(1, int(s["ticks_per_frame"]))
        hospital_cap = int(round(s["population"] * s["hospital_frac"]))

        sim = Simulation(WORLD_W, WORLD_H, seed=self.seed)
        sim.reset(population=s["population"], initial_infected=s["initial_infected"],
                  speed=s["speed"], distancing=s["distancing"],
                  vaccination=s["vaccination"], super_fraction=s["super_fraction"],
                  strains=s["strains"], cross_immunity=s.get("cross_immunity", 0.0))
        sim.isolate_infected = bool(s.get("isolate", False))
        renderer = ShortsRenderer(s["width"], s["height"], s["title"], renderer_meta(s))

        path = shorts_path(stamp, s.get("label"), idx + 1, total)
        writer = imageio.get_writer(
            path, fps=fps, codec="libx264", quality=8,
            pixelformat="yuv420p", macro_block_size=8,
            ffmpeg_params=["-movflags", "+faststart"])

        frames = 0
        over = 0
        epidemic_over = False
        abort = False
        last = None
        rt_text = None
        fade_frames = int(fps * 0.6)
        try:
            while frames < max_frames and (not epidemic_over or over < end_hold):
                if self._cancel:
                    abort = True
                    break
                if not epidemic_over:
                    for _ in range(tpf):
                        last = sim.step(s["radius"], s["strains"],
                                        s["waning_ticks"], hospital_cap)
                    if last[2] == 0 and last[3] == 0:
                        epidemic_over = True
                        over = 0
                else:
                    over += 1
                if frames % 10 == 0 or (epidemic_over and over == 0):
                    rt_text = _rt_text(sim.estimate_rt(s["strains"]),
                                       bool(s.get("showdown")))
                writer.append_data(renderer.render(
                    sim, epidemic_over, frame_idx=frames, max_frames=max_frames,
                    rt_text=rt_text, fade_frames=fade_frames))
                frames += 1
                if frames % 5 == 0:
                    self.progress.emit(idx, total, frames, max_frames)
        finally:
            try:
                writer.close()
            except Exception:
                if not abort:
                    raise

        if abort:
            try:
                os.remove(path)
            except OSError:
                pass
            return None

        if s.get("soundtrack"):
            try:
                _attach_soundtrack(path, _active_curve(sim, frames, fps, tpf), fps)
            except Exception:
                for junk in (path + ".wav", path + ".mux.mp4"):
                    try:
                        os.remove(junk)
                    except OSError:
                        pass

        self.progress.emit(idx, total, max_frames, max_frames)
        return path


# ---------------------------------------------------------------------------
# Head-to-head render thread
# ---------------------------------------------------------------------------
class SplitRenderThread(QThread):
    progress = Signal(int, int)
    completed = Signal(str, str)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            import imageio.v2 as imageio
        except ImportError:
            try:
                import imageio
            except ImportError:
                self.failed.emit(
                    "Video export needs imageio + its ffmpeg backend:\n"
                    "    pip install imageio imageio-ffmpeg")
                return
        sL, sR = self.cfg["sides"]
        fps = int(sL["fps"])
        max_frames = max(30, int(sL["max_seconds"] * fps))
        end_hold = int(fps * 2.5)
        tpf = max(1, int(sL["ticks_per_frame"]))
        seed = self.cfg["seed"]

        sims = []
        for s in (sL, sR):
            sim = Simulation(WORLD_W, WORLD_H, seed=seed)
            sim.reset(population=s["population"], initial_infected=s["initial_infected"],
                      speed=s["speed"], distancing=s["distancing"],
                      vaccination=s["vaccination"], super_fraction=s["super_fraction"],
                      strains=[s["strains"][0]])
            sim.isolate_infected = bool(s.get("isolate", False))
            sims.append(sim)
        sides_meta = [dict(label=s.get("label") or ("Left" if i == 0 else "Right"),
                           color=H2H_COLORS[i], radius=float(s["radius"]),
                           paramline=side_paramline(s))
                      for i, s in enumerate((sL, sR))]
        renderer = SplitRenderer(sL["width"], sL["height"], sL["title"],
                                 self.cfg["criterion_text"], sides_meta,
                                 show_map=bool(sL.get("show_map")))
        cap = int(round(sL["population"] * sL["hospital_frac"]))

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(
            _videos_dir(),
            f"virus_h2h_{stamp}_{_slugify(sides_meta[0]['label'])}"
            f"_vs_{_slugify(sides_meta[1]['label'])}.mp4")
        writer = imageio.get_writer(
            path, fps=fps, codec="libx264", quality=8,
            pixelformat="yuv420p", macro_block_size=8,
            ffmpeg_params=["-movflags", "+faststart"])

        frames = 0
        overs = [False, False]
        verdict = None
        verdict_at = None
        abort = False
        last = [None, None]
        rt_text = None
        fade_frames = int(fps * 0.6)
        try:
            while frames < max_frames:
                if self._cancel:
                    abort = True
                    break
                for i, sim in enumerate(sims):
                    if overs[i]:
                        continue
                    s = (sL, sR)[i]
                    for _ in range(tpf):
                        last[i] = sim.step(s["radius"], [s["strains"][0]],
                                           s["waning_ticks"], cap)
                    if last[i][2] == 0 and last[i][3] == 0:
                        overs[i] = True
                if frames % 10 == 0:
                    rt_text = _rt_text(sims[0].estimate_rt([sL["strains"][0]]),
                                       showdown=False)
                if overs[0] and overs[1] and verdict is None:
                    verdict = self._verdict(sims)
                    verdict_at = frames
                writer.append_data(renderer.render(
                    sims[0], sims[1], overs[0], overs[1], verdict,
                    frame_idx=frames, max_frames=max_frames,
                    rt_text=rt_text, fade_frames=fade_frames))
                frames += 1
                if frames % 5 == 0:
                    self.progress.emit(frames, max_frames)
                if verdict is not None and frames - verdict_at >= end_hold:
                    break
        finally:
            try:
                writer.close()
            except Exception:
                if not abort:
                    raise

        if abort:
            try:
                os.remove(path)
            except OSError:
                pass
            self.cancelled.emit()
            return

        if self.cfg.get("soundtrack"):
            try:
                act = 0.5 * (_active_curve(sims[0], frames, fps, tpf)
                             + _active_curve(sims[1], frames, fps, tpf))
                _attach_soundtrack(path, act, fps)
            except Exception:
                for junk in (path + ".wav", path + ".mux.mp4"):
                    try:
                        os.remove(junk)
                    except OSError:
                        pass

        self.progress.emit(max_frames, max_frames)
        if verdict is None:
            verdict = self._verdict(sims)
        st = verdict["stats"]
        wtxt = ("Draw" if verdict["winner"] < 0
                else f"{sides_meta[verdict['winner']]['label']} wins")
        summary = (f"{frames} frames @ {fps} fps ({frames / fps:.1f} s)\n"
                   f"Result: {wtxt} ({verdict['criterion']})\n"
                   f"Left:  {st[0]['infected']:,} infected · {st[0]['deaths']:,} dead · "
                   f"peak {st[0]['peak']:,} · day {st[0]['days']:.0f}\n"
                   f"Right: {st[1]['infected']:,} infected · {st[1]['deaths']:,} dead · "
                   f"peak {st[1]['peak']:,} · day {st[1]['days']:.0f}")
        self.completed.emit(path, summary)

    def _verdict(self, sims):
        stats = []
        for sim in sims:
            row = sim.history[-1]
            stats.append(dict(deaths=int(row[5]), infected=sim.ever_infected,
                              peak=int(sim.peak_infected),
                              days=float(row[0]) / TICKS_PER_DAY,
                              pop=sim.n_agents))
        key = self.cfg["criterion_key"]
        a, b = stats[0][key], stats[1][key]
        eps = 0.05 if key == "days" else 0.0
        if abs(a - b) <= eps:
            winner = -1
        else:
            winner = 0 if a < b else 1
        return dict(winner=winner, criterion=self.cfg["criterion_text"], stats=stats)


# ---------------------------------------------------------------------------
# Batch configuration dialog
# ---------------------------------------------------------------------------
SWEEP_PARAMS = (
    ("Transmission chance", "chance",       1, 100, "%",  lambda v: f"transmit {v}%"),
    ("Incubation period",   "incubation",   0,  14, "d",  lambda v: f"incubation {v}d"),
    ("Recovery time",       "recovery",     0, 100, "d",  lambda v: f"recovery {v}d" if v else "recovery never"),
    ("Mortality",           "mortality",    0,  60, "%",  lambda v: f"mortality {v}%"),
    ("Waning immunity",     "waning",       0, 360, "d",  lambda v: f"waning {v}d" if v else "immunity forever"),
    ("Vaccination",         "vaccination",  0,  90, "%",  lambda v: f"vaccination {v}%"),
    ("Super-spreaders",     "superspreader", 0, 20, "%", lambda v: f"super-spreaders {v}%"),
    ("Hospital capacity",   "hospitals",    2,  60, "%",  lambda v: f"hospitals {v}%"),
    ("Social distancing",   "distancing",   0,  90, "%",  lambda v: f"distancing {v}%"),
    ("Movement speed",      "speed",        0, 100, "",   lambda v: f"speed {v / 10:.1f}"),
    ("Infection radius",    "radius",       2,  25, "px", lambda v: f"radius {v}px"),
    ("Showdown · Alpha transmit",  "strainA_chance",    1, 100, "%", lambda v: f"Alpha transmit {v}%"),
    ("Showdown · Beta transmit",   "strainB_chance",    1, 100, "%", lambda v: f"Beta transmit {v}%"),
    ("Showdown · Gamma transmit",  "strainC_chance",    1, 100, "%", lambda v: f"Gamma transmit {v}%"),
    ("Showdown · Alpha mortality", "strainA_mortality", 0, 600, "‰", lambda v: f"Alpha mort {v / 10:g}%"),
    ("Showdown · Beta mortality",  "strainB_mortality", 0, 600, "‰", lambda v: f"Beta mort {v / 10:g}%"),
    ("Showdown · Gamma mortality", "strainC_mortality", 0, 600, "‰", lambda v: f"Gamma mort {v / 10:g}%"),
)
H2H_PARAMS = tuple(sp for sp in SWEEP_PARAMS if not sp[1].startswith("strain"))

STRAIN_SWEEP_IDX = {"A": 0, "B": 1, "C": 2}


def _apply_sweep(cfg, key, value):
    if key.startswith("strain") and len(key) > 6 and key[6] in STRAIN_SWEEP_IDX:
        idx = STRAIN_SWEEP_IDX[key[6]]
        field = "chance" if key.endswith("_chance") else "mortality"
        if idx < len(cfg["strains"]):
            cfg["strains"][idx][field] = (value / 100 if field == "chance"
                                          else value / 1000)
        return
    if key == "chance":
        cfg["strains"][0]["chance"] = value / 100
    elif key == "incubation":
        cfg["strains"][0]["incubation_ticks"] = value * TICKS_PER_DAY
    elif key == "recovery":
        cfg["strains"][0]["recovery_ticks"] = value * TICKS_PER_DAY
    elif key == "mortality":
        cfg["strains"][0]["mortality"] = value / 100
    elif key == "waning":
        cfg["waning_ticks"] = value * TICKS_PER_DAY
    elif key == "vaccination":
        cfg["vaccination"] = value / 100
    elif key == "superspreader":
        cfg["super_fraction"] = value / 100
    elif key == "hospitals":
        cfg["hospital_frac"] = value / 100
    elif key == "distancing":
        cfg["distancing"] = value / 100
    elif key == "speed":
        cfg["speed"] = value / 10
    elif key == "radius":
        cfg["radius"] = float(value)


class BatchDialog(QDialog):
    def __init__(self, main, parent=None):
        super().__init__(parent)
        self.main = main
        self.setWindowTitle("Batch render — parameter sweep")
        self.setMinimumWidth(460)

        v = QVBoxLayout(self)
        v.setSpacing(10)

        v.addWidget(QLabel("Base title"))
        self.title_edit = QLineEdit(main.short_title.text().strip()
                                    or "Virus Outbreak Simulation")
        v.addWidget(self.title_edit)

        grid = QGridLayout()
        self.param_combo = QComboBox()
        for name, *_rest in SWEEP_PARAMS:
            self.param_combo.addItem(name)
        grid.addWidget(QLabel("Sweep parameter"), 0, 0)
        grid.addWidget(self.param_combo, 0, 1, 1, 3)

        self.from_spin = QSpinBox()
        self.to_spin = QSpinBox()
        self.count_spin = QSpinBox()
        self.count_spin.setRange(2, 8)
        self.count_spin.setValue(5)
        grid.addWidget(QLabel("From"), 1, 0)
        grid.addWidget(self.from_spin, 1, 1)
        grid.addWidget(QLabel("To"), 1, 2)
        grid.addWidget(self.to_spin, 1, 3)
        grid.addWidget(QLabel("Variants"), 2, 0)
        grid.addWidget(self.count_spin, 2, 1)
        v.addLayout(grid)

        self.seed_box = QCheckBox("Same start conditions for every variant (shared seed)")
        self.seed_box.setChecked(True)
        v.addWidget(self.seed_box)

        self.info_label = QLabel()
        self.info_label.setObjectName("hint")
        self.info_label.setWordWrap(True)
        v.addWidget(self.info_label)

        v.addWidget(QLabel("Variants to render:"))
        self.list_widget = QListWidget()
        self.list_widget.setFixedHeight(150)
        v.addWidget(self.list_widget)

        btns = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        start_btn = QPushButton("Start batch render")
        start_btn.setObjectName("batchBtn")
        start_btn.clicked.connect(self.accept)
        btns.addWidget(cancel_btn)
        btns.addWidget(start_btn, 1)
        v.addLayout(btns)

        self.param_combo.currentIndexChanged.connect(self._param_changed)
        self.from_spin.valueChanged.connect(self._refresh)
        self.to_spin.valueChanged.connect(self._refresh)
        self.count_spin.valueChanged.connect(self._refresh)
        self._param_changed(0)

    def _param_changed(self, index):
        _name, _key, lo, hi, suffix, _fmt = SWEEP_PARAMS[index]
        for spin in (self.from_spin, self.to_spin):
            spin.setRange(lo, hi)
            spin.setSuffix(f" {suffix}".rstrip() if suffix else "")
        self.from_spin.setValue(lo)
        self.to_spin.setValue(hi)
        self._refresh()

    def _values(self):
        lo = self.from_spin.value()
        hi = self.to_spin.value()
        count = self.count_spin.value()
        if hi < lo:
            lo, hi = hi, lo
        raw = np.linspace(lo, hi, count)
        return sorted({int(round(x)) for x in raw})

    def _refresh(self):
        _name, _key, _lo, _hi, _suffix, fmt = SWEEP_PARAMS[self.param_combo.currentIndex()]
        vals = self._values()
        self.list_widget.clear()
        for v in vals:
            self.list_widget.addItem(fmt(v))
        s = self.main._shorts_settings()
        worst = s["max_seconds"] * s["fps"] * len(vals)
        self.info_label.setText(
            f'{s["width"]}×{s["height"]} · {s["fps"]} fps · up to {s["max_seconds"]} s each '
            f'(Shorts ≤ {MAX_SHORT_SECONDS} s)\n'
            f'Worst case ≈ {worst:,} frames total — big sweeps at 1080p/60fps take a while.\n'
            f'Showdown sweeps apply per strain · videos go to your Videos folder + manifest .txt.')

    def configs_and_seed(self):
        base = self.main._shorts_settings()
        _name, key, _lo, _hi, _suffix, fmt = SWEEP_PARAMS[self.param_combo.currentIndex()]
        title = self.title_edit.text().strip() or base["title"]
        seed = None
        if self.seed_box.isChecked():
            seed = int(np.random.default_rng().integers(0, 2 ** 31))
        configs = []
        for v in self._values():
            cfg = dict(base)
            cfg["strains"] = [dict(sp) for sp in base["strains"]]
            cfg["title"] = title
            cfg["label"] = fmt(v)
            _apply_sweep(cfg, key, v)
            configs.append(cfg)
        return configs, seed


# ---------------------------------------------------------------------------
# Head-to-head configuration dialog
# ---------------------------------------------------------------------------
class HeadToHeadDialog(QDialog):
    CRITERIA = (("Fewest deaths wins", "deaths"),
                ("Fewest infections wins", "infections"),
                ("Lowest peak wins", "peak"),
                ("Fastest burnout wins", "days"))

    def __init__(self, main, parent=None):
        super().__init__(parent)
        self.main = main
        self.setWindowTitle("Head-to-head render — split screen")
        self.setMinimumWidth(520)

        v = QVBoxLayout(self)
        v.setSpacing(10)

        v.addWidget(QLabel("Title"))
        self.title_edit = QLineEdit(main.short_title.text().strip()
                                    or "Virus Outbreak: Head-to-Head")
        v.addWidget(self.title_edit)

        self.side_widgets = []
        cols = QHBoxLayout()
        for i, name in enumerate(("Left", "Right")):
            g = QGroupBox(f"{name} side")
            gv = QVBoxLayout(g)
            gv.addWidget(QLabel("Label shown in video"))
            lbl = QLineEdit()
            gv.addWidget(lbl)
            gv.addWidget(QLabel("Override parameter"))
            combo = QComboBox()
            combo.addItem("— none (use current settings) —")
            for pname, *_rest in H2H_PARAMS:
                combo.addItem(pname)
            gv.addWidget(combo)
            gv.addWidget(QLabel("Override value"))
            spin = QSpinBox()
            spin.setEnabled(False)
            gv.addWidget(spin)
            self.side_widgets.append((lbl, combo, spin))
            cols.addWidget(g)
        v.addLayout(cols)

        row = QHBoxLayout()
        row.addWidget(QLabel("Win criterion"))
        self.criterion_combo = QComboBox()
        for text, _key in self.CRITERIA:
            self.criterion_combo.addItem(text)
        row.addWidget(self.criterion_combo, 1)
        v.addLayout(row)

        self.seed_box = QCheckBox("Shared seed — identical start, only the override differs")
        self.seed_box.setChecked(True)
        v.addWidget(self.seed_box)

        self.sound_box = QCheckBox("Add procedural soundtrack")
        v.addWidget(self.sound_box)

        self.info_label = QLabel()
        self.info_label.setObjectName("hint")
        self.info_label.setWordWrap(True)
        v.addWidget(self.info_label)

        btns = QHBoxLayout()
        preview_btn = QPushButton("Preview frame")
        preview_btn.clicked.connect(self.preview_frame)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        start_btn = QPushButton("Start head-to-head render")
        start_btn.setObjectName("h2hBtn")
        start_btn.clicked.connect(self.accept)
        btns.addWidget(preview_btn)
        btns.addWidget(cancel_btn)
        btns.addWidget(start_btn, 1)
        v.addLayout(btns)

        for i in range(2):
            lbl, combo, spin = self.side_widgets[i]
            combo.currentIndexChanged.connect(lambda _ix, s=i: self._param_changed(s))
            spin.valueChanged.connect(lambda _v, s=i: self._auto_label(s))

        dist_idx = next((j for j, sp in enumerate(H2H_PARAMS)
                         if sp[1] == "distancing"), -1)
        if dist_idx >= 0:
            _lbl, combo, spin = self.side_widgets[1]
            combo.setCurrentIndex(dist_idx + 1)
            spin.setValue(50)
        self._param_changed(0)

        s = main._shorts_settings()
        self.info_label.setText(
            f'{s["width"]}×{s["height"]} · {s["fps"]} fps · up to {s["max_seconds"]} s · '
            f'compares two single-virus runs (showdown mode not used here).\n'
            f'Whichever side finishes first freezes with an OUTBREAK OVER chip; '
            f'the verdict appears once both are done.')

    def _param_changed(self, side):
        lbl_edit, combo, spin = self.side_widgets[side]
        idx = combo.currentIndex()
        if idx <= 0:
            spin.setEnabled(False)
            lbl_edit.setText("Base")
        else:
            _name, _key, lo, hi, suffix, _fmt = H2H_PARAMS[idx - 1]
            spin.setRange(lo, hi)
            spin.setSuffix(f" {suffix}".rstrip() if suffix else "")
            spin.setValue((lo + hi) // 2)
            spin.setEnabled(True)
            self._auto_label(side)

    def _auto_label(self, side):
        lbl_edit, combo, spin = self.side_widgets[side]
        idx = combo.currentIndex()
        if idx <= 0:
            lbl_edit.setText("Base")
        else:
            _name, _key, _lo, _hi, _suffix, fmt = H2H_PARAMS[idx - 1]
            lbl_edit.setText(fmt(spin.value()))

    def side_cfg(self, i):
        base = self.main._shorts_settings()
        cfg = dict(base)
        cfg["strains"] = [dict(base["strains"][0])]
        cfg["title"] = self.title_edit.text().strip() or "Virus Outbreak: Head-to-Head"
        lbl_edit, combo, spin = self.side_widgets[i]
        idx = combo.currentIndex()
        label = lbl_edit.text().strip()
        if idx > 0:
            _name, key, _lo, _hi, _suffix, fmt = H2H_PARAMS[idx - 1]
            _apply_sweep(cfg, key, spin.value())
            if not label:
                label = fmt(spin.value())
        cfg["label"] = label or ("Left" if i == 0 else "Right")
        return cfg

    def render_config(self):
        cfgL = self.side_cfg(0)
        cfgR = self.side_cfg(1)
        text, key = self.CRITERIA[self.criterion_combo.currentIndex()]
        seed = None
        if self.seed_box.isChecked():
            seed = int(np.random.default_rng().integers(0, 2 ** 31))
        return dict(sides=[cfgL, cfgR], criterion_key=key,
                    criterion_text=text, seed=seed,
                    soundtrack=self.sound_box.isChecked())

    def preview_frame(self):
        cfg = self.render_config()
        cfgL, cfgR = cfg["sides"]
        sides_meta = [dict(label=c["label"], color=H2H_COLORS[i],
                           radius=float(c["radius"]), paramline=side_paramline(c))
                      for i, c in enumerate((cfgL, cfgR))]
        renderer = SplitRenderer(cfgL["width"], cfgL["height"], cfgL["title"],
                                 cfg["criterion_text"], sides_meta,
                                 show_map=bool(cfgL.get("show_map")))
        sim = self.main.sim
        arr = renderer.render(sim, sim, False, False, None)
        img = QImage(arr.tobytes(), arr.shape[1], arr.shape[0],
                     arr.shape[1] * 3, QImage.Format_RGB888)
        dlg = QDialog(self)
        dlg.setWindowTitle("Head-to-head preview — one frame")
        lay = QVBoxLayout(dlg)
        lbl = QLabel()
        pm = QPixmap.fromImage(img).scaledToHeight(
            min(700, self.main.height() - 100), Qt.SmoothTransformation)
        lbl.setPixmap(pm)
        lay.addWidget(lbl)
        close = QPushButton("Close")
        close.clicked.connect(dlg.accept)
        lay.addWidget(close)
        dlg.show()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Virus Lab — SEIR · Showdown · Head-to-Head · Interventions")
        self.resize(1280, 800)
        self.setMinimumSize(980, 640)

        self.params = {
            "speed": 3.0, "radius": 9.0, "chance": 0.10,
            "incubation_ticks": 2 * TICKS_PER_DAY,
            "recovery_ticks": 25 * TICKS_PER_DAY, "mortality": 0.08,
            "waning_ticks": 0, "super_fraction": 0.04, "hospital_frac": 0.15,
            "cross_immunity": 0.0, "steps_per_frame": 1,
        }
        self.sliders = {}
        self.epidemic_over = False
        self.render_thread = None
        self._applying_preset = False
        self._applying_matchup = False

        self.sim = Simulation(WORLD_W, WORLD_H)
        self.sim.reset(population=700, initial_infected=5,
                       speed=self.params["speed"], distancing=0.10,
                       super_fraction=self.params["super_fraction"])

        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.tick)

        self._build_ui()
        self._apply_chart_series()
        self.chart.set_data(self.sim.history, self.sim.n_agents)
        self.refresh_stats()
        self.timer.start()

    # -- UI construction -----------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        central.setObjectName("root")
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(8)

        stat_defs = (
            ("susceptible", "SUSCEPTIBLE", "#4da3ff"),
            ("exposed",     "EXPOSED",     "#f0883e"),
            ("infected",    "INFECTED",    "#ff5c5c"),
            ("recovered",   "RECOVERED",   "#3fb950"),
            ("dead",        "DEAD",        "#8b949e"),
            ("rt",          "RT (EST)",    "#58a6ff"),
            ("day",         "DAY",         "#e3b341"),
            ("peak",        "PEAK SICK",   "#d2a8ff"),
        )
        self.stat_vals = {}
        rows = (QHBoxLayout(), QHBoxLayout())
        for i, (key, title, color) in enumerate(stat_defs):
            rows[i // 4].setSpacing(8)
            box, val = self._stat_box(title, color)
            self.stat_vals[key] = val
            rows[i // 4].addWidget(box, 1)
        lv.addLayout(rows[0])
        lv.addLayout(rows[1])

        self.canvas = SimCanvas()
        self.canvas.attach(self.sim)
        self.canvas.infection_radius = self.params["radius"]
        self.canvas.hospital_frac = self.params["hospital_frac"]
        self.canvas.on_manual_infection = self.on_manual_infection
        lv.addWidget(self.canvas, 1)

        self.chart = ChartWidget()
        lv.addWidget(self.chart, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(self._build_controls())
        scroll.setMinimumWidth(340)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([900, 360])
        root.addWidget(splitter)

    def _stat_box(self, title, color):
        frame = QFrame()
        frame.setObjectName("statBox")
        frame.setStyleSheet(
            "QFrame#statBox { background: #161b22; border: 1px solid #21262d; "
            "border-radius: 10px; }"
        )
        v = QVBoxLayout(frame)
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(0)
        t = QLabel(title)
        t.setStyleSheet(f"color: {color}; font-size: 10px; font-weight: 600;")
        v.addWidget(t)
        val = QLabel("0")
        val.setObjectName("statValue")
        v.addWidget(val)
        return frame, val

    def _slider(self, layout, key, title, lo, hi, default, on_change, fmt=str):
        head = QHBoxLayout()
        name = QLabel(title)
        val = QLabel(fmt(default))
        val.setObjectName("sliderValue")
        head.addWidget(name)
        head.addStretch(1)
        head.addWidget(val)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(lo, hi)
        slider.setValue(default)
        layout.addLayout(head)
        layout.addWidget(slider)
        self.sliders[key] = slider

        def changed(value):
            val.setText(fmt(value))
            on_change(value)

        slider.valueChanged.connect(changed)

    def _build_controls(self):
        panel = QWidget()
        panel.setMinimumWidth(320)
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 4, 0)
        v.setSpacing(10)

        g_pre = QGroupBox("Virus preset (single-virus mode)")
        gp = QVBoxLayout(g_pre)
        self.preset_combo = QComboBox()
        for name in PRESETS:
            self.preset_combo.addItem(name)
        gp.addWidget(self.preset_combo)
        self.preset_desc = QLabel(PRESETS["Custom"]["desc"])
        self.preset_desc.setObjectName("hint")
        self.preset_desc.setWordWrap(True)
        gp.addWidget(self.preset_desc)
        v.addWidget(g_pre)
        self.preset_combo.currentIndexChanged.connect(self._apply_preset)

        g_pop = QGroupBox("Population & behavior")
        gp = QVBoxLayout(g_pop)
        self._slider(gp, "population", "Population (R)", 100, 2000, 700, lambda _x: None)
        self._slider(gp, "initial_infected", "Initially infected per strain (R)", 1, 40, 5,
                     lambda _x: None)
        self._slider(gp, "speed", "Movement speed", 0, 100, 30,
                     self._on_speed, fmt=lambda x: f"{x / 10:.1f} px/tick")
        self._slider(gp, "distancing", "Social distancing (R)", 0, 90, 10,
                     lambda _x: None, fmt=lambda x: f"{x} %")
        v.addWidget(g_pop)

        self.disease_group = QGroupBox("Disease (single-virus mode)")
        gd = QVBoxLayout(self.disease_group)
        self._slider(gd, "radius", "Infection radius", 2, 25, 9,
                     self._on_radius, fmt=lambda x: f"{x} px")
        self._slider(gd, "chance", "Transmission chance", 1, 100, 10,
                     self._on_chance, fmt=lambda x: f"{x} %")
        self._slider(gd, "incubation", "Incubation (exposed, not contagious)", 0, 14, 2,
                     self._on_incubation, fmt=lambda x: f"{x} d")
        self._slider(gd, "recovery", "Recovery time (0 = never)", 0, 100, 25,
                     self._on_recovery, fmt=lambda x: "never" if x == 0 else f"{x} days")
        self._slider(gd, "mortality", "Mortality", 0, 600, 80,
                     self._on_mortality, fmt=lambda x: f"{x / 10:g} %")
        v.addWidget(self.disease_group)

        v.addWidget(self._build_showdown_group())

        g_imm = QGroupBox("Immunity & carriers")
        gi = QVBoxLayout(g_imm)
        self._slider(gi, "vaccination", "Vaccination at start (R, all strains)", 0, 90, 0,
                     lambda _x: None, fmt=lambda x: f"{x} %")
        self._slider(gi, "waning", "Immunity lasts (0 = forever, single only)", 0, 360, 0,
                     self._on_waning, fmt=lambda x: "forever" if x == 0 else f"{x} d")
        self._slider(gi, "superspreader", "Super-spreaders (R, ×3 range)", 0, 20, 4,
                     self._on_superspreader, fmt=lambda x: f"{x} %")
        self._slider(gi, "hospitals", "Hospital capacity", 2, 60, 15,
                     self._on_hospitals, fmt=lambda x: f"{x} % of pop")
        v.addWidget(g_imm)

        g_int = QGroupBox("Interventions (live)")
        gint = QVBoxLayout(g_int)
        self._slider(gint, "vacc_now", "Vaccinate dose", 5, 60, 25,
                     lambda _x: None, fmt=lambda x: f"{x} %")
        irow = QHBoxLayout()
        self.vacc_btn = QPushButton("Vaccinate now (V)")
        self.vacc_btn.clicked.connect(self._do_vaccinate)
        self.lock_btn = QPushButton("Lockdown 14 d (L)")
        self.lock_btn.clicked.connect(self._toggle_lockdown)
        irow.addWidget(self.vacc_btn)
        irow.addWidget(self.lock_btn)
        gint.addLayout(irow)
        self.isolate_box = QCheckBox("Isolate infected (quarantine, instant)")
        self.isolate_box.toggled.connect(self._on_isolate)
        gint.addWidget(self.isolate_box)
        v.addWidget(g_int)

        g_run = QGroupBox("Playback")
        gr = QVBoxLayout(g_run)
        self._slider(gr, "steps", "Steps per frame", 1, 8, 1,
                     self._on_steps, fmt=lambda x: f"×{x}")
        self.map_live_box = QCheckBox("Conquest map overlay (territory by strain)")
        self.map_live_box.toggled.connect(self._toggle_map_live)
        gr.addWidget(self.map_live_box)
        csv_btn = QPushButton("Export curve CSV…")
        csv_btn.clicked.connect(self.export_csv)
        gr.addWidget(csv_btn)
        btns = QHBoxLayout()
        self.run_btn = QPushButton("Pause")
        self.run_btn.setObjectName("runBtn")
        self.run_btn.setFocusPolicy(Qt.NoFocus)
        self.run_btn.clicked.connect(self.toggle_run)
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setFocusPolicy(Qt.NoFocus)
        self.reset_btn.clicked.connect(self.reset_sim)
        btns.addWidget(self.run_btn, 1)
        btns.addWidget(self.reset_btn, 1)
        gr.addLayout(btns)
        v.addWidget(g_run)

        v.addWidget(self._build_shorts_group())

        hint = QLabel("(R) = applied on Reset · mortality ×4 while hospitals are\n"
                      "overwhelmed · recovery 0 = never recover · Rt > 1 growing,\n"
                      "< 1 shrinking · Space = pause · L = lockdown · V = vaccinate\n"
                      "· Click the world to seed the selected strain")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        v.addWidget(hint)

        self.status = QLabel("Running")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        v.addWidget(self.status)
        v.addStretch(1)
        return panel

    def _build_shorts_group(self):
        g = QGroupBox(f"YouTube Shorts (9:16 MP4, ≤ {MAX_SHORT_SECONDS} s)")
        v = QVBoxLayout(g)

        self.short_title = QLineEdit("Virus Outbreak Simulation")
        self.short_title.setPlaceholderText("Video title…")
        v.addWidget(self.short_title)

        row = QHBoxLayout()
        row.addWidget(QLabel("Size"))
        self.res_combo = QComboBox()
        self.res_combo.addItem("720 × 1280 (fast)", (720, 1280))
        self.res_combo.addItem("1080 × 1920 (HD)", (1080, 1920))
        row.addWidget(self.res_combo, 1)
        row.addWidget(QLabel("FPS"))
        self.fps_combo = QComboBox()
        for f in ("24", "30", "60"):
            self.fps_combo.addItem(f)
        self.fps_combo.setCurrentIndex(1)
        row.addWidget(self.fps_combo, 1)
        v.addLayout(row)

        self._slider(v, "dur", "Max length", 10, MAX_SHORT_SECONDS, 60,
                     lambda _x: None, fmt=lambda x: f"{x} s")
        self._slider(v, "tpf", "Render sim speed", 1, 10, 3,
                     lambda _x: None, fmt=lambda x: f"×{x} ticks/frame")

        self.sound_box = QCheckBox("Procedural soundtrack (intensity follows outbreak)")
        v.addWidget(self.sound_box)

        btns = QHBoxLayout()
        self.preview_btn = QPushButton("Preview")
        self.preview_btn.clicked.connect(self.preview_short)
        self.render_btn = QPushButton("Render MP4")
        self.render_btn.setObjectName("shortBtn")
        self.render_btn.clicked.connect(self.render_short)
        btns.addWidget(self.preview_btn)
        btns.addWidget(self.render_btn, 1)
        v.addLayout(btns)

        self.batch_btn = QPushButton("Batch render — parameter sweep…")
        self.batch_btn.setObjectName("batchBtn")
        self.batch_btn.clicked.connect(self.render_batch)
        v.addWidget(self.batch_btn)

        self.h2h_btn = QPushButton("Head-to-head render (split screen)…")
        self.h2h_btn.setObjectName("h2hBtn")
        self.h2h_btn.clicked.connect(self.render_h2h)
        v.addWidget(self.h2h_btn)

        self.map_video_box = QCheckBox("Conquest map overlay in video")
        v.addWidget(self.map_video_box)

        prog = QHBoxLayout()
        self.render_bar = QProgressBar()
        self.render_bar.hide()
        self.cancel_render_btn = QPushButton("Cancel")
        self.cancel_render_btn.hide()
        self.cancel_render_btn.clicked.connect(self.cancel_render)
        prog.addWidget(self.render_bar, 1)
        prog.addWidget(self.cancel_render_btn)
        v.addLayout(prog)
        return g

    # -- showdown group ----------------------------------------------------------
    def _build_showdown_group(self):
        g = QGroupBox("Multi-virus showdown")
        vg = QVBoxLayout(g)

        self.showdown_box = QCheckBox("Enable showdown (strains compete)")
        vg.addWidget(self.showdown_box)

        body = QWidget()
        vb = QVBoxLayout(body)
        vb.setContentsMargins(0, 0, 0, 0)
        vb.setSpacing(8)

        row = QHBoxLayout()
        row.addWidget(QLabel("Strains"))
        self.strain_count_combo = QComboBox()
        self.strain_count_combo.addItem("2")
        self.strain_count_combo.addItem("3")
        row.addWidget(self.strain_count_combo, 1)
        row.addWidget(QLabel("Matchup"))
        self.matchup_combo = QComboBox()
        for name in MATCHUPS:
            self.matchup_combo.addItem(name)
        row.addWidget(self.matchup_combo, 2)
        vb.addLayout(row)

        self._slider(vb, "cross", "Cross-immunity (0 = independent)", 0, 100, 0,
                     self._on_cross, fmt=lambda x: f"{x} %")

        self.strain_frames = []
        for i in range(3):
            frame = QGroupBox(STRAIN_NAMES[i])
            frame.setObjectName(f"strain{i}")
            fs = QVBoxLayout(frame)
            fs.setSpacing(6)
            self._slider(fs, f"s{i}_chance", "Transmit %/contact", 1, 100, 10,
                         self._on_strain_slider, fmt=lambda x: f"{x} %")
            self._slider(fs, f"s{i}_inc", "Incubation", 0, 14, 2,
                         self._on_strain_slider, fmt=lambda x: f"{x} d")
            self._slider(fs, f"s{i}_rec", "Recovery (0 = never)", 0, 100, 25,
                         self._on_strain_slider,
                         fmt=lambda x: "never" if x == 0 else f"{x} d")
            self._slider(fs, f"s{i}_mort", "Mortality", 0, 600, 80,
                         self._on_strain_slider, fmt=lambda x: f"{x / 10:g} %")
            vb.addWidget(frame)
            self.strain_frames.append(frame)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Click world infects:"))
        self.click_strain_combo = QComboBox()
        for name in STRAIN_NAMES:
            self.click_strain_combo.addItem(name)
        row2.addWidget(self.click_strain_combo, 1)
        vb.addLayout(row2)

        counts_row = QHBoxLayout()
        self.strain_count_labels = []
        for i, name in enumerate(STRAIN_NAMES):
            lbl = QLabel(f"{name}: 0")
            lbl.setStyleSheet(f"color: {STRAIN_COLORS[i].name()}; font-weight: bold;")
            counts_row.addWidget(lbl, 1)
            self.strain_count_labels.append(lbl)
        vb.addLayout(counts_row)

        vg.addWidget(body)
        body.setVisible(False)

        self.showdown_body = body
        self.showdown_box.toggled.connect(self._on_showdown_toggled)
        self.strain_count_combo.currentIndexChanged.connect(self._on_strain_count)
        self.matchup_combo.currentIndexChanged.connect(self._apply_matchup)
        self.click_strain_combo.currentIndexChanged.connect(
            lambda idx: setattr(self.canvas, "infect_strain", idx))
        return g

    def _strain_count(self):
        return self.strain_count_combo.currentIndex() + 2

    def _on_showdown_toggled(self, checked):
        self.showdown_body.setVisible(checked)
        self.disease_group.setEnabled(not checked)
        self._apply_chart_series()
        self.status.setText("Showdown armed — press Reset to seed all strains"
                            if checked else "Single-virus mode — Reset to apply")

    def _on_strain_count(self, _idx):
        n = self._strain_count()
        self.strain_frames[2].setVisible(n == 3)
        self.strain_count_labels[2].setVisible(n == 3)
        self._apply_chart_series()

    def _apply_matchup(self, _idx):
        name = self.matchup_combo.currentText()
        m = MATCHUPS.get(name)
        if not m:
            return
        self._applying_matchup = True
        try:
            self.strain_count_combo.setCurrentIndex(m["n"] - 2)
            for i, sp in enumerate(m["strains"]):
                self.sliders[f"s{i}_chance"].setValue(sp["chance"])
                self.sliders[f"s{i}_inc"].setValue(sp["inc"])
                self.sliders[f"s{i}_rec"].setValue(sp["rec"])
                self.sliders[f"s{i}_mort"].setValue(sp["mort"])
            self.sliders["cross"].setValue(m["cross"])
        finally:
            self._applying_matchup = False
        self.status.setText(f"Matchup applied: {name} — press Reset")

    def _sync_matchup_combo(self):
        if self._applying_matchup:
            return
        if self.matchup_combo.currentIndex() != 0:
            self.matchup_combo.blockSignals(True)
            self.matchup_combo.setCurrentIndex(0)
            self.matchup_combo.blockSignals(False)

    def _on_strain_slider(self, _v):
        self._sync_matchup_combo()

    def _on_cross(self, v):
        self.params["cross_immunity"] = v / 100
        self.sim.cross_immunity = v / 100

    def _apply_chart_series(self):
        if self.showdown_box.isChecked():
            n = self._strain_count()
            spec = [(1, STATE_COLORS[SUSCEPTIBLE], "Susceptible")]
            for i in range(n):
                spec.append((6 + i, STRAIN_COLORS[i], STRAIN_NAMES[i]))
            spec += [(4, STATE_COLORS[RECOVERED], "Recovered"),
                     (5, STATE_COLORS[DEAD], "Dead")]
        else:
            spec = [(1, STATE_COLORS[SUSCEPTIBLE], "Susceptible"),
                    (2, EXPOSED_COLOR, "Exposed"),
                    (3, STATE_COLORS[INFECTED], "Infected"),
                    (4, STATE_COLORS[RECOVERED], "Recovered"),
                    (5, STATE_COLORS[DEAD], "Dead")]
        self.chart.set_series(spec)

    # -- preset handling -----------------------------------------------------------
    def _apply_preset(self, index):
        name = self.preset_combo.currentText()
        if name not in PRESETS:
            return
        p = PRESETS[name]
        self._applying_preset = True
        try:
            self.sliders["chance"].setValue(p["chance"])
            self.sliders["radius"].setValue(p["radius"])
            self.sliders["incubation"].setValue(p["incubation"])
            self.sliders["recovery"].setValue(p["recovery"])
            self.sliders["mortality"].setValue(p["mortality"])
            self.sliders["superspreader"].setValue(p["superspreader"])
            self.sliders["waning"].setValue(p["waning"])
        finally:
            self._applying_preset = False
        self.preset_desc.setText(p["desc"])
        if name != "Custom":
            self.status.setText(f"Preset applied: {name}")

    def _sync_preset_combo(self):
        if self._applying_preset:
            return
        if self.preset_combo.currentIndex() != 0:
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentIndex(0)
            self.preset_combo.blockSignals(False)
            self.preset_desc.setText(PRESETS["Custom"]["desc"])

    # -- slider handlers --------------------------------------------------------------
    def _on_speed(self, v):
        self.params["speed"] = v / 10
        self.sim.set_speed(v / 10)

    def _on_radius(self, v):
        self.params["radius"] = float(v)
        self.canvas.infection_radius = float(v)
        self._sync_preset_combo()

    def _on_chance(self, v):
        self.params["chance"] = v / 100
        self._sync_preset_combo()

    def _on_incubation(self, v):
        self.params["incubation_ticks"] = v * TICKS_PER_DAY
        self._sync_preset_combo()

    def _on_recovery(self, v):
        self.params["recovery_ticks"] = v * TICKS_PER_DAY
        self._sync_preset_combo()

    def _on_mortality(self, v):
        self.params["mortality"] = v / 1000
        self._sync_preset_combo()

    def _on_waning(self, v):
        self.params["waning_ticks"] = v * TICKS_PER_DAY
        self._sync_preset_combo()

    def _on_superspreader(self, v):
        self.params["super_fraction"] = v / 100
        self._sync_preset_combo()

    def _on_hospitals(self, v):
        self.params["hospital_frac"] = v / 100
        self.canvas.hospital_frac = v / 100

    def _on_steps(self, v):
        self.params["steps_per_frame"] = int(v)

    def _toggle_map_live(self, checked):
        self.canvas.show_map = bool(checked)
        self.canvas.update()

    # -- interventions ------------------------------------------------------------
    def _do_vaccinate(self):
        k = self.sim.vaccinate_now(self.sliders["vacc_now"].value() / 100)
        self.refresh_stats()
        self.chart.set_data(self.sim.history, self.sim.n_agents)
        self.canvas.update()
        self.status.setText(f"Vaccinated {k:,} susceptible agents just now")

    def _toggle_lockdown(self):
        if self.sim.in_lockdown:
            self.sim.clear_lockdown()
            self.status.setText("Lockdown lifted — movement resumes")
        else:
            self.sim.set_lockdown(14)
            self.status.setText("Lockdown for 14 days — all movement stopped")
        self._update_lockdown_btn()
        self.canvas.update()

    def _update_lockdown_btn(self):
        if self.sim.in_lockdown:
            self.lock_btn.setText(f"End lockdown ({self.sim.lockdown_remaining_days:.0f} d)")
        else:
            self.lock_btn.setText("Lockdown 14 d (L)")

    def _on_isolate(self, checked):
        self.sim.isolate_infected = bool(checked)
        self.status.setText("Isolation ON — infected agents stop moving"
                            if checked else "Isolation OFF")

    # -- strain parameter bundle -------------------------------------------------------
    def _current_strains(self):
        if self.showdown_box.isChecked():
            out = []
            for i in range(self._strain_count()):
                out.append(dict(
                    chance=self.sliders[f"s{i}_chance"].value() / 100,
                    incubation_ticks=self.sliders[f"s{i}_inc"].value() * TICKS_PER_DAY,
                    recovery_ticks=self.sliders[f"s{i}_rec"].value() * TICKS_PER_DAY,
                    mortality=self.sliders[f"s{i}_mort"].value() / 1000,
                ))
            return out
        return [dict(chance=self.params["chance"],
                     incubation_ticks=self.params["incubation_ticks"],
                     recovery_ticks=self.params["recovery_ticks"],
                     mortality=self.params["mortality"])]

    # -- actions ---------------------------------------------------------------
    def toggle_run(self):
        if self.timer.isActive():
            self.timer.stop()
            self.run_btn.setText("Start")
            self.status.setText("Paused")
        else:
            self.timer.start()
            self.run_btn.setText("Pause")
            self.status.setText("Running")

    def reset_sim(self):
        sl = self.sliders
        showdown = self.showdown_box.isChecked()
        self.sim.reset(
            population=sl["population"].value(),
            initial_infected=sl["initial_infected"].value(),
            speed=self.params["speed"],
            distancing=sl["distancing"].value() / 100,
            vaccination=sl["vaccination"].value() / 100,
            super_fraction=sl["superspreader"].value() / 100,
            strains=self._current_strains(),
            cross_immunity=self.params["cross_immunity"] if showdown else 0.0,
        )
        self.epidemic_over = False
        self.canvas.infection_radius = self.params["radius"]
        self.chart.set_data(self.sim.history, self.sim.n_agents)
        self.refresh_stats()
        self.canvas.update()
        if not self.timer.isActive():
            self.timer.start()
        self.run_btn.setText("Pause")
        mode = f"showdown · {self.sim.n_strains} strains seeded" if showdown \
            else "Running — fresh outbreak seeded"
        self.status.setText(mode)

    def on_manual_infection(self):
        self.epidemic_over = False
        self.refresh_stats()
        self.chart.set_data(self.sim.history, self.sim.n_agents)
        self.canvas.update()
        if self.sim.showdown:
            name = STRAIN_NAMES[self.canvas.infect_strain]
            self.status.setText(f"Seeded an {name} outbreak — Start if paused")
        else:
            self.status.setText("Outbreak seeded — press Start if paused")

    def export_csv(self):
        hist = self.sim.history
        default = os.path.join(
            _videos_dir(),
            f"virus_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        path, _ = QFileDialog.getSaveFileName(self, "Export epidemic curve",
                                              default, "CSV (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["tick", "day", "susceptible", "exposed", "infected",
                            "recovered", "dead", "infected_alpha",
                            "infected_beta", "infected_gamma"])
                for row in hist:
                    w.writerow([int(row[0]), round(float(row[0]) / TICKS_PER_DAY, 2),
                                int(row[1]), int(row[2]), int(row[3]),
                                int(row[4]), int(row[5]), int(row[6]),
                                int(row[7]), int(row[8])])
            self.status.setText(f"Curve exported: {path}")
        except OSError as exc:
            QMessageBox.warning(self, "Export failed", str(exc))

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Space:
            self.toggle_run()
        elif e.key() == Qt.Key_L:
            self._toggle_lockdown()
        elif e.key() == Qt.Key_V:
            self._do_vaccinate()
        else:
            super().keyPressEvent(e)

    # -- per-frame update -----------------------------------------------------------
    def tick(self):
        p = self.params
        snap = None
        cap = int(round(self.sim.n_agents * p["hospital_frac"]))
        strains = self._current_strains()
        for _ in range(p["steps_per_frame"]):
            snap = self.sim.step(p["radius"], strains,
                                 p["waning_ticks"], cap)
            if snap[2] == 0 and snap[3] == 0:
                break
        if snap is None:
            return
        self.canvas.update()
        if self.sim.step_count % 2 == 0:
            self.chart.set_data(self.sim.history, self.sim.n_agents)
        self.refresh_stats()
        self._update_lockdown_btn()
        if snap[2] == 0 and snap[3] == 0 and not self.epidemic_over:
            pop = max(self.sim.n_agents, 1)
            if self.sim.ever_infected > snap[5]:
                self.epidemic_over = True
                self.chart.set_data(self.sim.history, self.sim.n_agents)
                self.timer.stop()
                self.run_btn.setText("Start")
                msg = (f"Epidemic over — {100 * self.sim.ever_infected / pop:.0f}% was "
                       f"infected at some point, {snap[5]} died "
                       f"({100 * snap[5] / pop:.1f}%).")
                if self.sim.showdown:
                    ever = self.sim.ever_per_strain
                    if ever.sum() > 0:
                        w = int(np.argmax(ever))
                        msg += f" {STRAIN_NAMES[w]} wins with {int(ever[w]):,} infections."
                self.status.setText(msg + " Press Reset to run again.")

    def refresh_stats(self):
        t, S, E, I, R, D, i0, i1, i2 = (int(x) for x in self.sim.history[-1])
        v = self.stat_vals
        v["susceptible"].setText(f"{S:,}")
        v["exposed"].setText(f"{E:,}")
        v["infected"].setText(f"{I:,}")
        v["recovered"].setText(f"{R:,}")
        v["dead"].setText(f"{D:,}")
        v["day"].setText(f"{t / TICKS_PER_DAY:.1f}")
        v["peak"].setText(f"{self.sim.peak_infected:,}")
        rt = self.sim.estimate_rt(self._current_strains())

        def fmt1(x):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return "–"
            if np.isinf(x):
                return "9.9+"
            return f"{x:.2f}"

        if self.showdown_box.isChecked() and rt is not None and len(rt) > 1:
            v["rt"].setText(" ".join(
                f"{STRAIN_GREEK[i]}{fmt1(rt[i])}"
                for i in range(min(len(rt), self.sim.n_strains))))
        else:
            v["rt"].setText(fmt1(rt[0]) if rt is not None and len(rt) else "–")
        if self.showdown_box.isChecked():
            for i, lbl in enumerate(self.strain_count_labels):
                lbl.setText(f"{STRAIN_NAMES[i]}: {(i0, i1, i2)[i]:,}")

    # -- YouTube Shorts --------------------------------------------------------------
    def _shorts_settings(self):
        w, h = self.res_combo.currentData()
        return dict(
            title=self.short_title.text().strip() or "Virus Outbreak Simulation",
            width=w, height=h,
            fps=int(self.fps_combo.currentText()),
            max_seconds=self.sliders["dur"].value(),
            ticks_per_frame=self.sliders["tpf"].value(),
            population=self.sliders["population"].value(),
            initial_infected=self.sliders["initial_infected"].value(),
            speed=self.params["speed"],
            distancing=self.sliders["distancing"].value() / 100,
            vaccination=self.sliders["vaccination"].value() / 100,
            super_fraction=self.params["super_fraction"],
            hospital_frac=self.params["hospital_frac"],
            radius=self.params["radius"],
            waning_ticks=self.params["waning_ticks"],
            isolate=self.isolate_box.isChecked(),
            soundtrack=self.sound_box.isChecked(),
            show_map=self.map_video_box.isChecked(),
            showdown=self.showdown_box.isChecked(),
            strains=self._current_strains(),
            cross_immunity=(self.params["cross_immunity"]
                            if self.showdown_box.isChecked() else 0.0),
        )

    def render_short(self):
        if not IMAGEIO_OK:
            self._warn_imageio()
            return
        if self.render_thread and self.render_thread.isRunning():
            return
        cfg = self._shorts_settings()
        cfg["strains"] = [dict(sp) for sp in cfg["strains"]]
        cfg["label"] = "showdown" if cfg["showdown"] else None
        self._start_render([cfg], seed=None)

    def render_batch(self):
        if self.render_thread and self.render_thread.isRunning():
            return
        if not IMAGEIO_OK:
            self._warn_imageio()
            return
        dlg = BatchDialog(self, self)
        if dlg.exec() != QDialog.Accepted:
            return
        configs, seed = dlg.configs_and_seed()
        if not configs:
            return
        self._start_render(configs, seed=seed)

    def render_h2h(self):
        if self.render_thread and self.render_thread.isRunning():
            return
        if not IMAGEIO_OK:
            self._warn_imageio()
            return
        dlg = HeadToHeadDialog(self, self)
        if dlg.exec() != QDialog.Accepted:
            return
        thread = SplitRenderThread(dlg.render_config(), parent=self)
        thread.progress.connect(self._on_split_progress)
        thread.completed.connect(self._on_h2h_done)
        thread.failed.connect(self._on_render_failed)
        thread.cancelled.connect(self._on_render_cancelled)
        self.render_thread = thread
        self._set_render_busy(True)
        self.render_bar.setRange(0, 1)
        self.render_bar.setValue(0)
        self.status.setText("Rendering head-to-head split-screen…")
        thread.start()

    def _start_render(self, configs, seed):
        self.render_thread = BatchRenderThread(configs, seed=seed, parent=self)
        self.render_thread.progress.connect(self._on_render_progress)
        self.render_thread.completed.connect(self._on_render_done)
        self.render_thread.failed.connect(self._on_render_failed)
        self.render_thread.cancelled.connect(self._on_render_cancelled)
        self._set_render_busy(True)
        self.render_bar.setRange(0, 1)
        self.render_bar.setValue(0)
        n = len(configs)
        seed_txt = f" · seed {seed}" if seed is not None else ""
        self.status.setText(f"Rendering {n} Short{'s' if n > 1 else ''}{seed_txt}…")
        self.render_thread.start()

    def cancel_render(self):
        if self.render_thread and self.render_thread.isRunning():
            self.render_thread.cancel()
            self.status.setText("Cancelling render…")

    def _warn_imageio(self):
        QMessageBox.warning(self, "Missing dependency",
                            "Video export needs imageio + its ffmpeg backend:\n\n"
                            "    pip install imageio imageio-ffmpeg")

    def preview_short(self):
        if self.render_thread and self.render_thread.isRunning():
            return
        s = self._shorts_settings()
        renderer = ShortsRenderer(s["width"], s["height"], s["title"], renderer_meta(s))
        rt_text = _rt_text(self.sim.estimate_rt(self._current_strains()),
                           bool(s["showdown"]))
        arr = renderer.render(self.sim, self.epidemic_over, rt_text=rt_text)
        img = QImage(arr.tobytes(), arr.shape[1], arr.shape[0],
                     arr.shape[1] * 3, QImage.Format_RGB888)

        dlg = QDialog(self)
        dlg.setWindowTitle("Short preview — one frame")
        lay = QVBoxLayout(dlg)
        lbl = QLabel()
        pm = QPixmap.fromImage(img).scaledToHeight(min(760, self.height() - 80),
                                                   Qt.SmoothTransformation)
        lbl.setPixmap(pm)
        lay.addWidget(lbl)
        info = QLabel(f'{s["width"]} × {s["height"]} · {s["fps"]} fps · '
                      f'up to {s["max_seconds"]} s')
        lay.addWidget(info)
        close = QPushButton("Close")
        close.clicked.connect(dlg.accept)
        lay.addWidget(close)
        dlg.show()

    def _set_render_busy(self, busy):
        self.render_btn.setEnabled(not busy)
        self.preview_btn.setEnabled(not busy)
        self.batch_btn.setEnabled(not busy)
        self.h2h_btn.setEnabled(not busy)
        self.render_bar.setVisible(busy)
        self.cancel_render_btn.setVisible(busy)
        if busy:
            self.render_bar.setValue(0)

    @Slot(int, int, int, int)
    def _on_render_progress(self, idx, total, frame, maxf):
        self.render_bar.setMaximum(max(maxf, 1))
        self.render_bar.setValue(frame)
        label = ""
        if self.render_thread and 0 <= idx < len(self.render_thread.configs):
            label = self.render_thread.configs[idx].get("label") or f"variant {idx + 1}"
        self.status.setText(f"Rendering {idx + 1}/{total}: {label}")

    @Slot(int, int)
    def _on_split_progress(self, frame, total):
        self.render_bar.setMaximum(max(total, 1))
        self.render_bar.setValue(frame)
        self.status.setText(f"Rendering head-to-head… {frame}/{total}")

    @Slot(object, str)
    def _on_render_done(self, results, manifest):
        self._set_render_busy(False)
        lines = [f"• {label}\n    {path}" for label, path in results]
        msg = f"Rendered {len(results)} video(s):\n\n" + "\n\n".join(lines)
        if manifest:
            msg += f"\n\nManifest: {manifest}"
        self.status.setText(f"Done — {len(results)} Short(s) saved")
        box = QMessageBox(self)
        box.setWindowTitle("Batch rendered")
        box.setText(msg)
        open_btn = box.addButton("Open folder", QMessageBox.ActionRole)
        box.addButton("OK", QMessageBox.AcceptRole)
        box.exec()
        if box.clickedButton() is open_btn and results:
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(os.path.dirname(os.path.abspath(results[0][1]))))

    @Slot(str, str)
    def _on_h2h_done(self, path, summary):
        self._set_render_busy(False)
        self.status.setText(f"Head-to-head saved: {path}")
        box = QMessageBox(self)
        box.setWindowTitle("Head-to-head rendered")
        box.setText(f"Your split-screen Short is ready:\n\n{path}\n\n{summary}")
        open_btn = box.addButton("Open folder", QMessageBox.ActionRole)
        box.addButton("OK", QMessageBox.AcceptRole)
        box.exec()
        if box.clickedButton() is open_btn:
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(os.path.dirname(os.path.abspath(path))))

    @Slot(str)
    def _on_render_failed(self, msg):
        self._set_render_busy(False)
        self.status.setText("Rendering failed")
        QMessageBox.critical(self, "Rendering failed", msg)

    @Slot()
    def _on_render_cancelled(self):
        self._set_render_busy(False)
        self.status.setText("Rendering cancelled")

    def closeEvent(self, e):
        if self.render_thread and self.render_thread.isRunning():
            self.render_thread.cancel()
            self.render_thread.wait(4000)
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()