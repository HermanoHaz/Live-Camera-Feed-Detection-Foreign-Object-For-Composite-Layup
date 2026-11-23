"""Detection engine: settings, camera discovery, ResNet-18 PatchCore model, FOD shape detection, worker thread.
No UI code in here."""
import os, sys, time
from datetime import datetime
import cv2, numpy as np, torch, torch.nn.functional as F, torchvision
from PySide6.QtCore import QThread, Signal


# ============================================================================
# SETTINGS (tune these)
# ============================================================================
IW, IH = 320, 240                      # network input -> 40x30 patch grid
GW, GH = IW // 8, IH // 8
SENS, CORESET_PCT, BASE_FRAMES = 0.7, 50, 6
MIN_THR = 1.0                          # threshold floor (a perfectly static feed would otherwise calibrate to ~0 and flag everything)
PIX_MIN, PIX_AREA = 25, 6              # pixel-level path: min brightness change and min size (half-res px) -> catches tiny/thin objects
FOD_COL = (94, 63, 244)                # BGR rose-red used for detected FOD (selected objects are blue)
REL_KEEP = 0.4                         # only patches >= 40% of the peak count as the object -> tight box, not the whole area
SHOW_HEAT = False                      # box only (set True to also draw a heatmap)
CAM_W, CAM_H = 1920, 1080              # camera is always requested at 1920x1080
ORIENT = "rotate"                      # landscape camera -> vertical: "rotate" (whole frame, 1080x1920) or "crop" (centre 9:16 strip)

os.makedirs("detections", exist_ok=True); os.makedirs("selected_fod", exist_ok=True)


# ============================================================================
# CAMERAS
# ============================================================================
try:
    from pygrabber.dshow_graph import FilterGraph
    HAS_PYG = sys.platform.startswith("win")
except Exception:
    HAS_PYG = False
BACKEND = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY


def list_cameras(active=None):
    """Returns [(opencv_index, name)]. Windows+pygrabber: real DirectShow names; otherwise probes indexes."""
    if HAS_PYG:
        try: return list(enumerate(FilterGraph().get_input_devices()))
        except Exception: pass
    out = []
    for i in range(8):
        if i == active: out.append((i, f"Camera {i}")); continue
        c = cv2.VideoCapture(i, BACKEND)
        if c.isOpened(): out.append((i, f"Camera {i}"))
        c.release()
    return out


# ============================================================================
# MODEL + IMAGE HELPERS
# ============================================================================
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_portrait(img, mode):
    """Always return a vertical (portrait) frame."""
    h, w = img.shape[:2]
    if h >= w: return img
    if mode == "rotate": return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    cw = int(h * 9 / 16); x = (w - cw) // 2
    return np.ascontiguousarray(img[:, x:x + cw])


def enhance(img):
    """CLAHE contrast on luminance + unsharp mask, so small debris stands out."""
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    l = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return cv2.addWeighted(out, 1.6, cv2.GaussianBlur(out, (0, 0), 2), -0.6, 0)


class PatchCore:
    def __init__(self):
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1      # downloads once, then cached
        self.m = torchvision.models.resnet18(weights=w).eval()
        self.bank = None

    @torch.no_grad()
    def features(self, bgr):
        rgb = cv2.cvtColor(cv2.resize(bgr, (IW, IH)), cv2.COLOR_BGR2RGB)
        x = (torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255 - MEAN) / STD
        m = self.m
        x = m.maxpool(m.relu(m.bn1(m.conv1(x)))); x = m.layer1(x)
        f2 = m.layer2(x); f3 = m.layer3(f2)
        f2 = F.avg_pool2d(f2, 3, 1, 1); f3 = F.avg_pool2d(f3, 3, 1, 1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3], 1)[0].permute(1, 2, 0).reshape(-1, 384)   # (1200, 384)

    def fit(self, frames, pct):
        x = torch.cat([self.features(f) for f in frames])
        k = max(500, int(len(x) * pct / 100))
        if k >= len(x):
            self.bank = x
        else:                                       # greedy k-center coreset on a random projection
            z = x @ (torch.randn(384, 128) / 128 ** .5)
            idx = [int(torch.randint(len(x), (1,)))]
            md = torch.cdist(z, z[idx[0]:idx[0] + 1]).squeeze(1)
            for _ in range(k - 1):
                i = int(md.argmax()); idx.append(i)
                md = torch.minimum(md, torch.cdist(z, z[i:i + 1]).squeeze(1))
            self.bank = x[idx]
        return len(self.bank)

    @torch.no_grad()
    def score_map(self, bgr):
        s = torch.cdist(self.features(bgr), self.bank).min(1).values.reshape(GH, GW).numpy()
        return cv2.GaussianBlur(s, (0, 0), 1.0)


# ============================================================================
# FOD SHAPES (pixel-level detection + drawing)
# ============================================================================
class DetectionMixin:
    """Methods used by the Worker thread: verdict, pixel-level objects, drawing."""

    def verdict(self, img, sm, mx):
        thr = max(self.base * (2 - SENS), MIN_THR); out, bad, cnt = self.overlay(img, sm, thr)
        ratio = max(mx / thr, self.pix_ratio)                  # tiny objects show up in the pixel path, not the patch score
        return out, bad, dict(state="DEFECT" if bad else "CLEAR", thr=thr, cnt=cnt,
                              score=min(100, int(ratio * 50)), peak=min(100., ratio * 50))

    @staticmethod
    def draw_mark(out, n, x0, y0, x1, y1):
        c = (248, 189, 56)                                 # blue = user-selected object (detected FOD boxes are red)
        cv2.rectangle(out, (x0, y0), (x1, y1), c, 2)
        tw = 14 + 10 * len(str(n)); ty = y0 - 22 if y0 >= 22 else y0
        cv2.rectangle(out, (x0, ty), (x0 + tw, ty + 22), c, -1)
        cv2.putText(out, str(n), (x0 + 7, ty + 16), cv2.FONT_HERSHEY_SIMPLEX, .55, (20, 12, 6), 2)

    @staticmethod
    def gs(a):                                             # half-resolution grey, lightly blurred
        return cv2.GaussianBlur(cv2.cvtColor(cv2.resize(a, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY), (0, 0), 1.0)

    def pixdiff(self, img):
        """Brightness-compensated difference vs the learned clean surface (half resolution)."""
        if self.base_gray is None: return None
        g = self.gs(img).astype(np.float32); b = self.base_gray.astype(np.float32)
        if g.shape != b.shape: return None
        dd = g - b
        return np.clip(np.abs(dd - np.median(dd)), 0, 255).astype(np.uint8)

    @property
    def pix_thr(self): return max(PIX_MIN, 1.6 * self.pix_noise)

    def objects(self, d, roi):
        """Exact pixel shape of every FOD (d = pixel difference vs the clean surface). Works for tiny/thin objects too.
        Separate objects (gap between them) stay separate; anything attached is ONE object.
        Returns [(outer_contour, [hole_contours])] at full-res, or None."""
        sh, sw = d.shape; S = .5
        m = d > self.pix_thr                               # pixel path: finds objects the patch AI is too coarse to see
        if roi is not None:                                # patch-flagged regions: slightly more sensitive local threshold
            r = cv2.dilate(cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_NEAREST), np.ones((max(3, sh // GH), max(3, sw // GW)), np.uint8))
            v = d[r > 0]
            if v.size and v.max() >= 25:
                t = max(20, .8 * self.pix_thr, cv2.threshold(v.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
                m = m | ((d > t) & (r > 0))
        m = cv2.morphologyEx(m.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))   # fills holes, never bridges two objects
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        out = []
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < PIX_AREA: continue
            cs, hr = cv2.findContours((lab == i).astype(np.uint8) * 255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if not cs: continue
            outer = max((c for c, hh in zip(cs, hr[0]) if hh[3] == -1), key=cv2.contourArea)
            holes = [c for c, hh in zip(cs, hr[0]) if hh[3] != -1 and cv2.contourArea(c) >= 30]    # blank areas inside the shape
            out.append(((outer / S).astype(np.int32), [(c / S).astype(np.int32) for c in holes]))
        return out or None

    @staticmethod
    def fod_tag(out, x, y):                                # small "FOD" label sitting on the shape's highest point
        ty = y - 22 if y >= 22 else y
        cv2.rectangle(out, (x, ty), (x + 62, ty + 22), FOD_COL, -1)
        cv2.putText(out, "FOD", (x + 6, ty + 16), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)

    def fod_box(self, out, x0, y0, x1, y1):                # fallback only (no pixel shape available)
        cv2.rectangle(out, (x0, y0), (x1, y1), FOD_COL, 2); self.fod_tag(out, x0, y0)

    def overlay(self, img, sm, thr):
        h, w = img.shape[:2]; r = sm / thr; pk = float(r.max())
        keep = max(1.0, REL_KEEP * pk)                         # keep only the peak region around the object
        rd = np.where(r >= keep, r, 0).astype(np.float32); out = img.copy()
        if self.show_heat and pk >= 1:
            heat = np.clip(cv2.resize(rd, (w, h), interpolation=cv2.INTER_CUBIC), 0, None)
            a = np.clip(heat / pk, 0, 1)[..., None] * .7
            col = cv2.applyColorMap(np.clip(heat / pk * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
            out = (img * (1 - a) + col * a).astype(np.uint8)
        mask = (rd > 0).astype(np.uint8)
        cnt = int(mask.sum()); bad = False
        d = self.pixdiff(img)
        shapes = self.objects(d, (r >= 1).astype(np.uint8) if cnt else None) if d is not None else None
        self.pix_ratio = float(d.max()) / self.pix_thr if shapes else 0.0
        if shapes:                                              # highlight ONLY the object's exact shape (no rectangle)
            fm = np.zeros((h, w), np.uint8)                      # fill mask: the object only, blank holes left clear
            for c, holes in sorted(shapes, key=lambda t: -cv2.contourArea(t[0])):
                cv2.drawContours(fm, [c], -1, 255, -1)
                if holes: cv2.drawContours(fm, holes, -1, 0, -1)
            ov = out.copy(); ov[fm > 0] = FOD_COL
            out = cv2.addWeighted(ov, .3, out, .7, 0)
            for c, holes in shapes:
                cv2.drawContours(out, [c] + holes, -1, FOD_COL, 2)
                x, y = c[c[:, 0, 1].argmin(), 0]; self.fod_tag(out, int(x), int(y))
            bad = True
        else:                                                   # fallback (no clean picture / no pixel change): patch boxes
            for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                if cv2.contourArea(c) + len(c) < 3: continue    # ignore single-patch noise
                x, y, bw, bh = cv2.boundingRect(c); bad = True
                self.fod_box(out, int(x * w / GW), int(y * h / GH), int((x + bw) * w / GW), int((y + bh) * h / GH))
        return out, bad, cnt


# ============================================================================
# WORKER THREAD (camera -> learn -> calibrate -> detect)
# ============================================================================
class Worker(DetectionMixin, QThread):
    """Processes the newest camera frame: learn clean surface -> calibrate -> detect."""
    frame = Signal(object, dict)

    def __init__(self):
        super().__init__()
        self.run_flag, self.last, self.reset_req = True, None, False
        self.orient = ORIENT
        self.show_heat = SHOW_HEAT
        self.enhance, self.marks = False, []              # enhancer on/off, user-selected boxes to draw
        self.want = self.cur = None                      # camera index requested / currently open


    def reset(self):
        self.buf, self.t_buf, self.calib, self.base, self.was_bad = [], 0.0, [], 0.0, False
        self.base_img = self.base_gray = None             # averaged clean surface (colour + half-res grey) for pixel-level detection
        self.pix_noise = self.pix_ratio = 0.0             # camera noise level measured while calibrating
        self.core.bank = None; self.reset_req = False

    def run(self):
        self.frame.emit(None, {"state": "LOADING MODEL"})
        self.core = PatchCore(); self.reset(); cap = None
        while self.run_flag:
            if self.want != self.cur:                                   # camera switched
                if cap: cap.release(); cap = None
                self.cur = self.want; self.last = None; self.reset()
                if self.cur is not None:
                    cap = cv2.VideoCapture(self.cur, BACKEND)
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))     # MJPG gives full fps at 1080p
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
                    if not cap.isOpened(): cap = None; self.frame.emit(None, {"state": "CAMERA ERROR"})
            if cap is None: time.sleep(.05); continue
            ok, img = cap.read()
            if not ok: time.sleep(.02); continue
            img = to_portrait(img, self.orient)
            if self.enhance: img = enhance(img)
            self.last = img
            if self.reset_req: self.reset()
            st = {"res": f"{img.shape[1]}x{img.shape[0]}", "state": "LEARNING"}; out = img
            if self.core.bank is None:                                  # 1) learn clean surface
                if time.time() - self.t_buf > .3:
                    self.buf.append(img.copy()); self.t_buf = time.time(); st["ref"] = self.buf[-1]   # reference box shows frames as they are learned
                st["learned"] = len(self.buf)
                if len(self.buf) >= BASE_FRAMES:
                    st["bank"] = self.core.fit(self.buf, CORESET_PCT); st["ref"] = self.buf[-1]
                    self.base_img = np.mean(self.buf, axis=0, dtype=np.float32).astype(np.uint8)
                    self.base_gray = self.gs(self.base_img); self.buf = []
            else:
                t0 = time.time(); sm = self.core.score_map(img); mx = float(sm.max())
                st.update(ms=int((time.time() - t0) * 1000), dist=mx, bank=len(self.core.bank))
                if len(self.calib) < 20:                                # 2) calibrate
                    self.calib.append(mx); self.base = max(self.calib); st["state"] = "CALIBRATING"
                    d0 = self.pixdiff(img)                               # learn how noisy the camera is at pixel level
                    if d0 is not None: self.pix_noise = max(self.pix_noise, float(np.percentile(d0, 99.95)))
                else:                                                   # 3) detect
                    out, bad, vs = self.verdict(img, sm, mx); st.update(vs)
                    if bad and not self.was_bad:
                        cv2.imwrite(f"detections/fod_{datetime.now():%Y%m%d_%H%M%S}.jpg", out)
                    self.was_bad = bad
            if self.marks:                                      # selected objects stay on the live feed
                if out is img: out = img.copy()
                for n, (x0, y0, x1, y1) in enumerate(self.marks, 1): self.draw_mark(out, n, x0, y0, x1, y1)
            self.frame.emit(out, st)
        if cap: cap.release()