import os
import cv2
import numpy as np

from config import MATCH_THRESHOLD_FAR, MATCH_THRESHOLD_NEAR, GAP_THRESHOLD_FAR, GAP_THRESHOLD_NEAR

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec
    return vec / norm

def compute_cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
    feat1 = l2_normalize(feat1)
    feat2 = l2_normalize(feat2)
    n1 = np.linalg.norm(feat1)
    n2 = np.linalg.norm(feat2)
    if n1 < 1e-8 or n2 < 1e-8:
        return -1.0
    return float(np.dot(feat1, feat2))

def point_in_box(cx, cy, box) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= cx <= x2 and y1 <= cy <= y2

def make_label(name: str, score: float) -> str:
    return f"{name} ({score * 100:.0f}%)" if score > 0 else name

def get_face_threshold(face_w: int) -> float:
    if face_w < 40:
        return MATCH_THRESHOLD_FAR
    if face_w > 80:
        return MATCH_THRESHOLD_NEAR
    ratio = (face_w - 40) / 40.0
    return MATCH_THRESHOLD_FAR + (MATCH_THRESHOLD_NEAR - MATCH_THRESHOLD_FAR) * ratio

def get_gap_threshold(face_w: int) -> float:
    if face_w < 45:
        return GAP_THRESHOLD_FAR
    if face_w > 85:
        return GAP_THRESHOLD_NEAR
    ratio = (face_w - 45) / 40.0
    return GAP_THRESHOLD_FAR + (GAP_THRESHOLD_NEAR - GAP_THRESHOLD_FAR) * ratio

def enhance_face_image(img: np.ndarray) -> np.ndarray:
    """
    Tiền xử lý tập trung cho ảnh khuôn mặt trước khi trích xuất embedding:
    - Cân bằng sáng (CLAHE) để giảm thiểu ảnh hưởng của ánh sáng.
    - Resize ảnh nếu quá nhỏ (ArcFace khuyên dùng 112x112).
    """
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

        target_size = 112
        h, w = enhanced.shape[:2]
        if w < target_size or h < target_size:
            scale = max(target_size / w, target_size / h)
            enhanced = cv2.resize(enhanced, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
        return enhanced
    except Exception:
        return img

def get_entrance_roi(frame_w: int, frame_h: int):
    roi_w = int(frame_w * 0.60)
    roi_h = int(frame_h * 0.78)
    x1 = (frame_w - roi_w) // 2
    y1 = int(frame_h * 0.15)
    x2 = x1 + roi_w
    y2 = y1 + roi_h
    return (x1, y1, x2, y2)

def crop_with_padding(frame: np.ndarray, box, pad_ratio: float) -> np.ndarray:
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    bw = x2 - x1
    bh = y2 - y1

    pad_w = int(bw * pad_ratio)
    pad_h = int(bh * pad_ratio)

    x1p = max(0, x1 - pad_w)
    y1p = max(0, y1 - pad_h)
    x2p = min(w, x2 + pad_w)
    y2p = min(h, y2 + pad_h)

    crop = frame[y1p:y2p, x1p:x2p]
    return crop.copy() if crop.size > 0 else np.empty((0, 0, 3), dtype=np.uint8)
