"""Nara_Images — DIP interactive exam prep.

Run:  python -m uvicorn app:app --reload   (from the backend/ directory)
or just double-click run.bat in the project root.
"""
import base64
import os
import re
import time

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Response, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dip.common import calc_hist, is_color, to_gray
from registry import BY_ID, CHAPTERS, public_catalog

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES = os.path.join(ROOT, "images")
CODES = os.path.join(ROOT, "codes")
FRONTEND = os.path.join(ROOT, "frontend")
UPLOADS = os.path.join(ROOT, "images", "_uploads")

os.makedirs(UPLOADS, exist_ok=True)

app = FastAPI(title="Nara_Images")

MAX_SIDE = 720  # keep the per-pixel loops and the round trip snappy


class Op(BaseModel):
    fn: str
    params: dict = {}


class ApplyRequest(BaseModel):
    image: str
    ops: list[Op] = []


def _safe_image_path(name):
    """Resolve `name` inside images/ and refuse anything that escapes it."""
    path = os.path.abspath(os.path.join(IMAGES, name))
    if not path.startswith(os.path.abspath(IMAGES)) or not os.path.isfile(path):
        raise HTTPException(404, f"ไม่พบรูป: {name}")
    return path


def _read(name):
    img = cv2.imread(_safe_image_path(name), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(400, f"อ่านรูปไม่ได้: {name}")
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def _png_b64(img):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(500, "เข้ารหัส PNG ไม่สำเร็จ")
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def _hist_payload(img):
    """Histogram for the panel: one series for gray, three for colour."""
    if is_color(img):
        return {"เขียว": calc_hist(img[:, :, 1]).tolist(),
                "น้ำเงิน": calc_hist(img[:, :, 0]).tolist(),
                "แดง": calc_hist(img[:, :, 2]).tolist()}
    return {"ความถี่": calc_hist(img).tolist()}


@app.get("/api/functions")
def get_functions():
    return {"chapters": CHAPTERS, "functions": public_catalog()}


@app.get("/api/images")
def get_images():
    out = []
    for name in sorted(os.listdir(IMAGES)):
        path = os.path.join(IMAGES, name)
        if not os.path.isfile(path):
            continue
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        h, w = img.shape[:2]
        out.append({"name": name, "w": int(w), "h": int(h),
                    "color": img.ndim == 3, "uploaded": False})
    for name in sorted(os.listdir(UPLOADS)):
        path = os.path.join(UPLOADS, name)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        h, w = img.shape[:2]
        out.append({"name": "_uploads/" + name, "w": int(w), "h": int(h),
                    "color": img.ndim == 3, "uploaded": True})
    return {"images": out}


@app.get("/api/thumb/{name:path}")
def get_thumb(name: str):
    """Always re-encode to PNG — browsers cannot display the .tif test images."""
    img = cv2.imread(_safe_image_path(name), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(400, f"อ่านรูปไม่ได้: {name}")
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    h, w = img.shape[:2]
    s = 180 / max(h, w)
    if s < 1:
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(500, "เข้ารหัส thumbnail ไม่สำเร็จ")
    return Response(content=buf.tobytes(), media_type="image/png",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.post("/api/upload")
async def upload(file: UploadFile):
    raw = await file.read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(400, "ไฟล์นี้ไม่ใช่รูปภาพที่อ่านได้")
    stem, ext = os.path.splitext(file.filename or "image")
    if ext.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        ext = ".png"
    # keep the user's own filename so the UI shows something recognisable,
    # but strip anything that could escape UPLOADS or confuse the path
    stem = re.sub(r"[^A-Za-z0-9ก-๙_.-]", "_", stem).strip("._") or "image"
    stem = stem[:60]

    name = stem + ext
    n = 1
    while os.path.exists(os.path.join(UPLOADS, name)):
        name = "%s_%d%s" % (stem, n, ext)
        n += 1

    cv2.imwrite(os.path.join(UPLOADS, name), img)
    return {"name": "_uploads/" + name}


@app.get("/api/code/{fn_id}")
def get_code(fn_id: str):
    spec = BY_ID.get(fn_id)
    if not spec:
        raise HTTPException(404, f"ไม่รู้จักฟังก์ชัน: {fn_id}")
    base = spec.get("code", fn_id)
    out = {}
    for variant in ("loop", "cv2"):
        path = os.path.join(CODES, f"{base}_{variant}.py")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                out[variant] = fh.read()
    return {"code": out}


@app.post("/api/apply")
def apply(req: ApplyRequest):
    img = _read(req.image)
    original = img.copy()
    hist_before = _hist_payload(img)

    steps, applied = [], []
    for op in req.ops:
        spec = BY_ID.get(op.fn)
        if not spec:
            raise HTTPException(404, f"ไม่รู้จักฟังก์ชัน: {op.fn}")

        need = spec.get("input", "any")
        if need == "gray" and is_color(img):
            img = to_gray(img)
        if need == "color" and not is_color(img):
            raise HTTPException(
                400,
                f"'{spec['name']}' ต้องใช้ภาพสี แต่ภาพตอนนี้เป็นภาพเทาแล้ว"
                + (" — ฟังก์ชันก่อนหน้าในไปป์ไลน์แปลงเป็นเทาไปแล้ว ลองสลับลำดับดู"
                   if applied else " ลองเลือกรูปสี"),
            )

        params = {}
        for p in spec.get("params", []):
            if p["key"] in op.params and op.params[p["key"]] is not None:
                params[p["key"]] = op.params[p["key"]]

        t0 = time.perf_counter()
        try:
            img, step = spec["fn"](img, **params)
        except HTTPException:
            raise
        except Exception as exc:  # surface the real error instead of a blank 500
            raise HTTPException(500, f"'{spec['name']}' ทำงานผิดพลาด: {exc}")
        ms = (time.perf_counter() - t0) * 1000

        applied.append({"fn": op.fn, "name": spec["name"], "params": params, "ms": round(ms, 1)})
        if step is not None:
            d = step.to_dict()
            d["fn"] = op.fn
            steps.append(d)

    return {
        "before": _png_b64(original),
        "after": _png_b64(img),
        "hist_before": hist_before,
        "hist_after": _hist_payload(img),
        "steps": steps,
        "applied": applied,
        "shape": {"h": int(img.shape[0]), "w": int(img.shape[1]),
                  "color": bool(is_color(img))},
    }


app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
