"""All the UI: stylesheet, custom widgets and the main window."""
from datetime import datetime
import cv2, numpy as np
from PySide6.QtCore import Qt, Signal, QSize, QRect, QPoint, QRectF, QTimer
from PySide6.QtGui import QPainter, QPen, QColor, QFont, QPixmap, QPainterPath, QTextOption, QImage
from PySide6.QtWidgets import (QWidget, QLabel, QPushButton, QFrame, QComboBox, QRubberBand, QSizePolicy,
                               QVBoxLayout, QHBoxLayout, QGridLayout)
from engine import GW, GH, BASE_FRAMES, HAS_PYG, list_cameras, Worker


# ============================================================================
# THEME (change the look here)
# ============================================================================
QSS = """
* { font-family: "Inter", "Segoe UI Variable Text", "Segoe UI", Arial; font-size: 13px; color: #e6eaf2; }
QWidget#root { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #07090f, stop:1 #0d121c); }
QLabel { background: transparent; }

/* cards */
QFrame#card { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #141925, stop:1 #0f131b);
              border: 1px solid #1d2431; border-radius: 16px; }
QFrame#row { background: transparent; border: 0; border-bottom: 1px solid #1a2130; }

/* typography */
QLabel#logo { font-size: 24px; font-weight: 800; color: #f5a524; }
QLabel#title { font-size: 17px; font-weight: 700; }
QLabel#muted, QLabel#hint { color: #8b94a7; }
QLabel#hint { font-size: 12px; }
QLabel#sec { color: #8b94a7; font-size: 11px; font-weight: 700; }
QLabel#k { color: #8b94a7; font-size: 12px; }
QLabel#v { font-weight: 600; }
QLabel#state { font-size: 13px; font-weight: 800; color: #8b94a7; }
QLabel#state[tone="ok"] { color: #34d399; }
QLabel#state[tone="bad"] { color: #fb7185; }
QLabel#state[tone="warn"] { color: #f5a524; }

/* chips / pills */
QLabel#chip { background: #171d29; border: 1px solid #222b3a; border-radius: 12px; padding: 4px 12px; font-size: 12px; font-weight: 700; color: #aab3c5; }
QLabel#chip[tone="ok"]   { color: #34d399; background: rgba(52, 211, 153, 12%);  border-color: rgba(52, 211, 153, 38%); }
QLabel#chip[tone="bad"]  { color: #fb7185; background: rgba(251, 113, 133, 12%); border-color: rgba(251, 113, 133, 40%); }
QLabel#chip[tone="warn"] { color: #f5a524; background: rgba(245, 165, 36, 12%);  border-color: rgba(245, 165, 36, 38%); }

/* alert box */
QFrame#alert { background: #141a25; border: 1px solid #222b3a; border-radius: 12px; }
QFrame#alert[tone="ok"]   { background: rgba(52, 211, 153, 8%);  border-color: rgba(52, 211, 153, 32%); }
QFrame#alert[tone="bad"]  { background: rgba(251, 113, 133, 9%); border-color: rgba(251, 113, 133, 36%); }
QFrame#alert[tone="warn"] { background: rgba(245, 165, 36, 8%);  border-color: rgba(245, 165, 36, 32%); }
QLabel#aicon { min-width: 26px; max-width: 26px; min-height: 26px; max-height: 26px; border-radius: 13px;
               background: #222b3a; color: #aab3c5; font-weight: 800; }
QLabel#aicon[tone="info"] { background: #38bdf8; color: #04202e; }
QLabel#aicon[tone="ok"]   { background: #34d399; color: #04281c; }
QLabel#aicon[tone="bad"]  { background: #fb7185; color: #2b0a12; }
QLabel#aicon[tone="warn"] { background: #f5a524; color: #2b1a03; }
QLabel#atext { color: #cbd3e1; }

/* buttons */
QPushButton { background: #171d29; border: 1px solid #222b3a; border-radius: 10px; padding: 10px 14px;
              font-weight: 600; outline: none; }
QPushButton:hover { background: #1d2535; border-color: #34425b; }
QPushButton:pressed { background: #131923; }
QPushButton:checked { background: rgba(245, 165, 36, 14%); border: 1px solid #f5a524; color: #f5a524; }
QPushButton#primary { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #f5a524, stop:1 #ff7a18);
                      color: #1a1205; border: 0; font-weight: 700; }
QPushButton#primary:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ffb43a, stop:1 #ff8a30); }
QPushButton#tx { background: rgba(251, 113, 133, 16%); color: #fb7185; border: 0; border-radius: 6px; padding: 0; font-size: 11px; font-weight: 800; }
QPushButton#tx:hover { background: #fb7185; color: #ffffff; }

/* camera picker */
QComboBox { background: #171d29; border: 1px solid #222b3a; border-radius: 10px; padding: 10px 36px 10px 12px; font-weight: 600; }
QComboBox:hover { border-color: #34425b; }
QComboBox::drop-down { border: 0; width: 0px; }
QComboBox QAbstractItemView { background: #131824; border: 1px solid #222b3a; border-radius: 8px; padding: 4px;
                              selection-background-color: #1f2838; selection-color: #e6eaf2; outline: 0; }

/* video views (border colour shows the state) */
QLabel#view { background: #05070b; border: 2px solid #1d2431; border-radius: 14px; color: #566074; font-size: 14px; }
QLabel#view[tone="ok"]   { border-color: #34d399; }
QLabel#view[tone="bad"]  { border-color: #fb7185; }
QLabel#view[tone="warn"] { border-color: #f5a524; }

/* selected-object tiles */
QFrame#thumb { background: #0c1018; border: 1px solid #38bdf8; border-radius: 10px; }
QLabel#tnum { background: #38bdf8; color: #04202e; font-weight: 800; border-top-left-radius: 9px; border-bottom-right-radius: 9px; }
"""


# ============================================================================
# WIDGETS
# ============================================================================
def repolish(w):
    w.style().unpolish(w); w.style().polish(w); w.update()


def tone(w, t):
    """Set the 'tone' property used by the stylesheet (ok / bad / warn / info / '')."""
    if w.property("tone") != t: w.setProperty("tone", t); repolish(w)


def label(text="", name=None, wrap=False, spacing=0.0):
    l = QLabel(text)
    if name: l.setObjectName(name)
    if wrap: l.setWordWrap(True)
    if spacing: f = l.font(); f.setLetterSpacing(QFont.AbsoluteSpacing, spacing); l.setFont(f)
    return l


def card():
    f = QFrame(); f.setObjectName("card"); return f


def rounded(pm, r=12):
    """Round the corners of a frame so it sits cleanly inside the rounded border."""
    out = QPixmap(pm.size()); out.fill(Qt.transparent)
    p = QPainter(out); p.setRenderHint(QPainter.Antialiasing)
    path = QPainterPath(); path.addRoundedRect(QRectF(out.rect()), r, r); p.setClipPath(path); p.drawPixmap(0, 0, pm); p.end()
    return out


class Combo(QComboBox):
    """Combo box with a drawn chevron (so the stylesheet can stay clean)."""
    def paintEvent(self, e):
        super().paintEvent(e)
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor("#8b94a7"), 1.8); pen.setCapStyle(Qt.RoundCap); p.setPen(pen)
        x, y = self.width() - 22, self.height() // 2
        p.drawLine(x - 4, y - 2, x, y + 2); p.drawLine(x, y + 2, x + 4, y - 2)


class ScoreRing(QWidget):
    """Circular gauge: 0-100 anomaly score, coloured by state."""
    COL = {"idle": "#2a3345", "ok": "#34d399", "bad": "#fb7185", "warn": "#f5a524"}

    def __init__(self):
        super().__init__(); self.setMinimumHeight(190); self.val, self.txt, self.t = 0, "--", "idle"

    def set(self, val, txt, t):
        self.val, self.txt, self.t = val, txt, t; self.update()

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        s = min(self.width(), self.height()) - 26
        r = QRectF((self.width() - s) / 2, (self.height() - s) / 2, s, s)
        pen = QPen(QColor("#1b2230"), 12); pen.setCapStyle(Qt.RoundCap); p.setPen(pen); p.drawArc(r, 0, 360 * 16)
        col = QColor(self.COL.get(self.t, self.COL["idle"]))
        if self.val > 0:
            pen.setColor(col); p.setPen(pen); p.drawArc(r, 90 * 16, -int(min(self.val, 100) / 100 * 360 * 16))
        f = QFont(self.font()); f.setPixelSize(int(s * .30)); f.setWeight(QFont.ExtraBold); p.setFont(f)
        p.setPen(QColor("#e6eaf2") if self.t != "idle" else QColor("#566074"))
        p.drawText(r.adjusted(0, -s * .05, 0, -s * .05), self.txt, QTextOption(Qt.AlignCenter))
        f.setPixelSize(11); f.setWeight(QFont.DemiBold); f.setLetterSpacing(QFont.AbsoluteSpacing, 1.4); p.setFont(f)
        p.setPen(QColor("#8b94a7")); p.drawText(r.adjusted(0, s * .36, 0, s * .36), "ANOMALY SCORE", QTextOption(Qt.AlignCenter))


class LiveView(QLabel):
    """Live feed. Click-hold-drag draws a box; release emits the box in image coordinates."""
    selected = Signal(int, int, int, int)

    def __init__(self, text):
        super().__init__(text); self.setObjectName("view"); self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(380, 285); self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.img_w = self.img_h = 0; self.o = None; self.live = False; self.sc, self.ox, self.oy = 1.0, 0.0, 0.0
        self.band = QRubberBand(QRubberBand.Rectangle, self)

    def show_frame(self, pm, w, h, sc, ox, oy):
        self.live = True; self.img_w, self.img_h = w, h; self.sc, self.ox, self.oy = sc, ox, oy; self.setPixmap(pm); self.setCursor(Qt.CrossCursor)

    def box(self, p, shift):
        dx, dy = p.x() - self.o.x(), p.y() - self.o.y()
        if shift:
            s = max(abs(dx), abs(dy)); dx = s if dx >= 0 else -s; dy = s if dy >= 0 else -s
        return QRect(self.o, QPoint(self.o.x() + dx, self.o.y() + dy)).normalized()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self.live:
            self.o = e.position().toPoint(); self.band.setGeometry(QRect(self.o, QSize())); self.band.show()

    def mouseMoveEvent(self, e):
        if self.o is not None:
            self.band.setGeometry(self.box(e.position().toPoint(), False))

    def mouseReleaseEvent(self, e):
        if self.o is None: return
        r = self.box(e.position().toPoint(), False); self.o = None; self.band.hide()
        pm = self.pixmap()
        if pm is None or pm.isNull() or r.width() < 8 or r.height() < 8: return
        cr = self.contentsRect(); s = self.sc                      # map box -> camera pixels (feed is shown "cover")
        cl = lambda v, m: max(0, min(m, int(v)))
        x0, x1 = cl(self.ox + (r.left() - cr.x()) / s, self.img_w), cl(self.ox + (r.right() + 1 - cr.x()) / s, self.img_w)
        y0, y1 = cl(self.oy + (r.top() - cr.y()) / s, self.img_h), cl(self.oy + (r.bottom() + 1 - cr.y()) / s, self.img_h)
        if x1 - x0 > 4 and y1 - y0 > 4: self.selected.emit(x0, y0, x1, y1)


class Thumb(QFrame):
    """Selected-object tile: number tab in the top-left corner, 'x' (on hover) in the top-right, picture below."""
    removed = Signal()

    def __init__(self, n, pm):
        super().__init__(); self.setObjectName("thumb"); self.setFixedSize(68, 86)
        t = QLabel(str(n), self); t.setObjectName("tnum"); t.setAlignment(Qt.AlignCenter); t.setGeometry(1, 1, 24, 16)
        im = QLabel(self); im.setPixmap(pm); im.setAlignment(Qt.AlignCenter); im.setGeometry(3, 21, 62, 62)
        self.x = QPushButton("✕", self); self.x.setObjectName("tx"); self.x.setGeometry(47, 1, 20, 16)
        self.x.setFocusPolicy(Qt.NoFocus); self.x.setToolTip("Remove"); self.x.hide(); self.x.clicked.connect(self.removed)

    def enterEvent(self, e): self.x.show()
    def leaveEvent(self, e): self.x.hide()


# ============================================================================
# MAIN WINDOW
# ============================================================================
STATS = ["Distance score (L2)", "Decision threshold", "Inference time", "Stream resolution",
         "Reference bank", "Peak intensity", "Affected patches"]


def box(parent_layout_cls, margins=(16, 14, 16, 16), spacing=10):
    """A card with a layout inside. Returns (card, layout)."""
    c = card(); l = parent_layout_cls(c); l.setContentsMargins(*margins); l.setSpacing(spacing); return c, l


class MainWindow(QWidget):
    def __init__(self):
        super().__init__(); self.setObjectName("root"); self.setStyleSheet(QSS)
        self.setWindowTitle("VOID · FOD Detection / AI Vision"); self.resize(1500, 860)
        self.w = Worker(); self.w.frame.connect(self.on_frame); self.w.start()
        self.cam, self.devs, self.ids, self.sel, self.ref_img = None, [], None, [], None     # cam = active camera index

        root = QVBoxLayout(self); root.setContentsMargins(24, 18, 24, 24); root.setSpacing(18)
        root.addLayout(self.build_header())
        body = QHBoxLayout(); body.setSpacing(18); root.addLayout(body, 1)
        body.addWidget(self.build_left()); body.addLayout(self.build_center(), 1); body.addWidget(self.build_right())
        for b in self.findChildren(QPushButton): b.setFocusPolicy(Qt.NoFocus)         # no dotted focus box after clicking
        if HAS_PYG:                                                                   # auto-rescan for new cameras
            self.tm = QTimer(self); self.tm.timeout.connect(self.refresh_devices); self.tm.start(3000)
        self.refresh_devices(True); self.draw_crops(); self.reset_ui()

    # ------------------------------------------------------------------ layout
    def build_header(self):
        h = QHBoxLayout(); h.setSpacing(14)
        h.addWidget(label("VOID", "logo", spacing=4))
        sep = QFrame(); sep.setFixedSize(1, 28); sep.setStyleSheet("background:#222b3a"); h.addWidget(sep)
        t = QVBoxLayout(); t.setSpacing(0)
        t.addWidget(label("FOD Detection", "title")); t.addWidget(label("AI vision · foreign object debris inspection", "muted"))
        h.addLayout(t); h.addStretch(1)
        self.pill = label("● NO CAMERA", "chip"); h.addWidget(self.pill)
        return h

    def build_left(self):
        col = QWidget(); col.setFixedWidth(300); v = QVBoxLayout(col); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(16)

        c, l = box(QVBoxLayout)                                                       # camera
        l.addWidget(label("CAMERA", "sec", spacing=1.2))
        self.combo = Combo(); self.combo.activated.connect(self.use_camera); l.addWidget(self.combo)
        v.addWidget(c)

        s, l = box(QVBoxLayout)                                                       # selected objects
        l.addWidget(label("SELECTED FOREIGN OBJECT", "sec", spacing=1.2))
        l.addWidget(label("Click and hold on the live feed, drag a box around the object, then release.", "hint", wrap=True))
        self.empty = label("Nothing selected yet.", "hint"); l.addWidget(self.empty)
        holder = QWidget(); self.grid = QGridLayout(holder); self.grid.setSpacing(10); self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setAlignment(Qt.AlignTop | Qt.AlignLeft); l.addWidget(holder); l.addStretch(1)
        v.addWidget(s, 1)

        k, l = box(QVBoxLayout)                                                       # controls
        l.addWidget(label("CONTROLS", "sec", spacing=1.2))
        fs = QPushButton("Full Screen"); fs.clicked.connect(lambda: self.showFullScreen())
        xs = QPushButton("Exit Full Screen"); xs.clicked.connect(lambda: self.showNormal())
        r = QHBoxLayout(); r.setSpacing(8); r.addWidget(fs); r.addWidget(xs); l.addLayout(r)
        rl = QPushButton("↻  Re-learn Surface"); rl.setObjectName("primary")
        rl.setToolTip("Learn the clean surface again (keep the ply clear of debris)"); rl.clicked.connect(lambda: self.relearn())
        be = QPushButton("✦  Image Enhancer"); be.setCheckable(True); be.setToolTip("Contrast + sharpen so debris stands out")
        be.toggled.connect(self.enhance_toggled)
        l.addWidget(rl); l.addWidget(be)
        v.addWidget(k)
        return col

    def view_card(self, chip, hint, view):
        c, l = box(QVBoxLayout, (14, 12, 14, 14), 10)
        top = QHBoxLayout(); top.addWidget(chip); top.addStretch(1); top.addWidget(label(hint, "hint")); l.addLayout(top)
        l.addWidget(view, 1); return c

    def build_center(self):
        row = QHBoxLayout(); row.setSpacing(18)
        self.vr = label("Waiting for camera", "view"); self.vr.setAlignment(Qt.AlignCenter)
        self.vr.setMinimumSize(380, 285); self.vr.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.vl = LiveView("No camera selected"); self.vl.selected.connect(self.add_crop)
        chip_r = label("● GOOD REFERENCE", "chip"); tone(chip_r, "ok"); self.chip_l = label("● LIVE FEED", "chip")
        row.addWidget(self.view_card(chip_r, "auto-learned clean surface", self.vr), 1)
        row.addWidget(self.view_card(self.chip_l, "drag to mark an object", self.vl), 1)
        return row

    def build_right(self):
        c, v = box(QVBoxLayout, (18, 16, 18, 16), 6); c.setFixedWidth(300)
        v.addWidget(label("DETECTION", "sec", spacing=1.2))
        self.ring = ScoreRing(); v.addWidget(self.ring)
        self.stt = label("NO CAMERA", "state", spacing=1); self.stt.setAlignment(Qt.AlignCenter); v.addWidget(self.stt); v.addSpacing(6)
        self.kv = {}
        for name in STATS:
            row = QFrame(); row.setObjectName("row"); h = QHBoxLayout(row); h.setContentsMargins(0, 8, 0, 8)
            val = label("–", "v"); val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            h.addWidget(label(name, "k")); h.addStretch(1); h.addWidget(val); v.addWidget(row); self.kv[name] = val
        v.addSpacing(8)
        self.alert = QFrame(); self.alert.setObjectName("alert"); h = QHBoxLayout(self.alert); h.setContentsMargins(12, 12, 12, 12); h.setSpacing(10)
        self.aic = label("i", "aicon"); self.aic.setAlignment(Qt.AlignCenter); self.al = label("", "atext", wrap=True)
        h.addWidget(self.aic, 0, Qt.AlignTop); h.addWidget(self.al, 1); v.addWidget(self.alert); v.addStretch(1)
        v.addWidget(label("Detections are saved to ./detections", "hint"))
        return c

    # ------------------------------------------------------------------ small UI helpers
    def set_alert(self, t, icon, text):
        tone(self.alert, t); tone(self.aic, t); self.aic.setText(icon); self.al.setText(text)

    def set_state(self, text, t): self.stt.setText(text); tone(self.stt, t)
    def set_pill(self, text, t): self.pill.setText(text); tone(self.pill, t)

    def reset_ui(self):                                    # idle look (no camera)
        self.vl.live = False; self.vl.unsetCursor(); self.vl.setText("No camera selected"); self.vr.setText("Waiting for camera")
        for w_ in (self.vl, self.vr, self.chip_l): tone(w_, "")
        self.ring.set(0, "--", "idle"); self.set_state("NO CAMERA", ""); self.set_pill("● NO CAMERA", "")
        self.set_alert("info", "i", "No camera selected. Pick one from the Camera card.")
        for val in self.kv.values(): val.setText("–")

    # ------------------------------------------------------------------ camera
    def refresh_devices(self, force=False):
        if self.ids is not None and not force and not HAS_PYG: return
        cams = list_cameras(self.cam); ids = [i for i, _ in cams]
        if ids == self.ids and not force: return
        cur = self.cam; self.devs, self.ids = cams, ids
        self.combo.blockSignals(True); self.combo.clear(); self.combo.addItem("No camera")
        for _, n in cams: self.combo.addItem(n)
        self.combo.blockSignals(False)
        if cur in ids: self.combo.setCurrentIndex(ids.index(cur) + 1)       # same camera keeps running
        else:
            self.combo.setCurrentIndex(0)                                   # starts on "No camera"
            if cur is not None: self.stop_camera()                          # active camera was unplugged

    def stop_camera(self):
        self.clear_sel(); self.cam = None; self.w.want = None; self.ref_img = None; self.reset_ui()

    def use_camera(self, i):
        if i <= 0 or i > len(self.devs): return self.stop_camera()
        self.clear_sel(); self.cam = self.devs[i - 1][0]; self.w.want = self.cam; self.relearn()

    def relearn(self):
        if self.cam is None: return
        self.w.reset_req = True; self.ref_img = None
        self.vr.setText("Learning clean surface…"); tone(self.vr, ""); self.ring.set(0, "…", "warn")
        self.set_alert("warn", "…", "Starting camera…")

    def enhance_toggled(self, on):
        self.w.enhance = on
        if self.cam is not None: self.relearn()                             # surface must be re-learned with the same filter

    # ------------------------------------------------------------------ selected foreign objects
    def add_crop(self, x0, y0, x1, y1):
        img = self.w.last
        if img is None: return
        crop = img[y0:y1, x0:x1].copy()
        if crop.size == 0: return
        cv2.imwrite(f"selected_fod/fod_{datetime.now():%Y%m%d_%H%M%S_%f}.png", crop)
        self.sel.append(((x0, y0, x1, y1), crop)); del self.sel[:-12]; self.draw_crops()

    def del_crop(self, n):
        del self.sel[n]; self.draw_crops()

    def clear_sel(self):                                   # positions only make sense for the current camera/orientation
        self.sel = []; self.draw_crops()

    def draw_crops(self):
        self.w.marks = [b for b, _ in self.sel]            # worker draws these as numbered blue boxes on the live feed
        while self.grid.count(): self.grid.takeAt(0).widget().deleteLater()
        self.empty.setVisible(not self.sel)
        for n, (_, c) in enumerate(self.sel):
            t = Thumb(n + 1, self.pix(c, 62, 62)); t.removed.connect(lambda n=n: self.del_crop(n))   # also removes it from the live feed
            self.grid.addWidget(t, n // 3, n % 3)

    # ------------------------------------------------------------------ display
    @staticmethod
    def pix(bgr, w, h):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); hh, ww = rgb.shape[:2]
        q = QImage(rgb.data, ww, hh, ww * 3, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(q).scaled(max(w, 1), max(h, 1), Qt.KeepAspectRatio, Qt.SmoothTransformation)

    @staticmethod
    def cover(bgr, cw, ch):
        """Fill the whole box (no blank bars): scale to cover, centre-crop the overflow."""
        cw, ch = max(cw, 1), max(ch, 1); h, w = bgr.shape[:2]; s = max(cw / w, ch / h)
        vw, vh = cw / s, ch / s; ox, oy = (w - vw) / 2, (h - vh) / 2
        crop = bgr[int(round(oy)):int(round(oy + vh)), int(round(ox)):int(round(ox + vw))]
        rgb = cv2.cvtColor(np.ascontiguousarray(crop), cv2.COLOR_BGR2RGB); hh, ww = rgb.shape[:2]
        q = QImage(rgb.data, ww, hh, ww * 3, QImage.Format_RGB888).copy()
        return rounded(QPixmap.fromImage(q).scaled(cw, ch, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)), s, ox, oy

    def show_ref(self):
        if self.ref_img is not None:
            cr = self.vr.contentsRect(); self.vr.setPixmap(self.cover(self.ref_img, cr.width(), cr.height())[0]); tone(self.vr, "ok")

    def resizeEvent(self, e): super().resizeEvent(e); self.show_ref(); QTimer.singleShot(0, self.show_ref)

    def on_frame(self, img, st):
        if self.cam is None: return                                          # ignore late results after 'No camera'
        k, s = self.kv, st["state"]
        if img is not None:
            cr = self.vl.contentsRect(); pm, sc, ox, oy = self.cover(img, cr.width(), cr.height())
            self.vl.show_frame(pm, img.shape[1], img.shape[0], sc, ox, oy)
            res = st["res"].replace("x", " × "); k["Stream resolution"].setText(res); self.set_pill("● LIVE  ·  " + res, "ok")
        if "ref" in st: self.ref_img = st["ref"]; self.show_ref()
        if "bank" in st: k["Reference bank"].setText(f"{st['bank']} patches")
        if "ms" in st: k["Inference time"].setText(f"{st['ms']} ms"); k["Distance score (L2)"].setText(f"{st['dist']:.4f}")
        if s in ("DEFECT", "CLEAR"):
            bad = s == "DEFECT"; t = "bad" if bad else "ok"
            self.ring.set(st["score"], str(st["score"]), t); self.set_state("DEFECT DETECTED" if bad else "SURFACE CLEAR", t)
            k["Decision threshold"].setText(f"{st['thr']:.4f}"); k["Peak intensity"].setText(f"{st['peak']:.1f}%")
            k["Affected patches"].setText(f"{st['cnt']} / {GW * GH}")
            self.set_alert(t, "!" if bad else "✓", "FOD detected – remove debris before placing the next ply." if bad
                           else "No debris found. Safe to place the next ply.")
            tone(self.vl, t); tone(self.chip_l, t)
        elif s in ("CALIBRATING", "LEARNING"):
            self.ring.set(0, "…", "warn"); self.set_state("CALIBRATING" if s == "CALIBRATING" else "LEARNING SURFACE", "warn")
            self.set_alert("warn", "…", "Calibrating – keep the clean surface in view." if s == "CALIBRATING" else
                           f"Learning the clean surface ({st.get('learned', 0)}/{BASE_FRAMES}) – keep the ply clear of debris.")
            tone(self.vl, "warn"); tone(self.chip_l, "warn")
        elif s == "CAMERA ERROR":
            self.set_pill("● CAMERA ERROR", "bad"); self.set_state("CAMERA ERROR", "bad")
            self.set_alert("bad", "!", "Could not open this camera. It may be in use by another app (close Discord/OBS preview using it) – then pick it again.")
        elif s == "LOADING MODEL":
            self.set_state("LOADING MODEL", "warn"); self.set_alert("info", "i", "Loading AI model (first run downloads the ResNet-18 weights)…")

    def closeEvent(self, e):
        self.w.want = None; self.w.run_flag = False; self.w.wait(); e.accept()