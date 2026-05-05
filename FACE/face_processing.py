import os
import cv2
import math
import pickle
import hashlib
import numpy as np
from typing import Dict, List, Optional, Tuple
from deepface import DeepFace
import mediapipe as mp

from config import (
    FACE_LANDMARKER_MODEL, MIN_FACE_SIZE, MIN_FACE_BLUR_REALTIME,
    MAX_YAW_ANGLE, MAX_PITCH_ANGLE, MIN_BRIGHTNESS, MAX_BRIGHTNESS,
    MIN_CONTRAST, INTRA_CLASS_MIN_SIM, FACE_PADDING_RATIO,
    EMBED_MODEL_NAME, UNKNOWN_MIN_SIM, CACHE_FILE
)
from utils import (
    l2_normalize, compute_cosine_similarity, get_face_threshold,
    get_gap_threshold, enhance_face_image, crop_with_padding
)

# Mediapipe FaceLandmarker (khởi tạo lazy)
_face_landmarker_instance = None
SHOW_DEBUG = True  # Can be imported from config, but keeping here for simple debug

def compute_blur_score(img: np.ndarray) -> float:
    if img is None or img.size == 0:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def compute_face_quality(face_crop: np.ndarray, recog_score: float) -> float:
    if face_crop is None or face_crop.size == 0:
        return -1.0

    h, w = face_crop.shape[:2]
    area_score = min((h * w) / 15000.0, 2.0)
    blur_score = min(compute_blur_score(face_crop) / 150.0, 3.0)
    recog_bonus = max(0.0, recog_score) * 4.0
    return area_score + blur_score + recog_bonus

def get_face_landmarker():
    """Lazy initialization cho Mediapipe FaceLandmarker (chỉ tạo 1 lần)"""
    global _face_landmarker_instance
    if _face_landmarker_instance is None:
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=FACE_LANDMARKER_MODEL
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
        )
        _face_landmarker_instance = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    return _face_landmarker_instance

def align_face(face_crop: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float]]]:
    if face_crop is None or face_crop.size == 0:
        return None, None

    h, w = face_crop.shape[:2]
    try:
        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        landmarker = get_face_landmarker()
        result = landmarker.detect(mp_image)

        if not result.face_landmarks:
            return face_crop, None

        landmarks = result.face_landmarks[0]
        left_eye = landmarks[33]
        right_eye = landmarks[263]

        left_eye_pt = (left_eye.x * w, left_eye.y * h)
        right_eye_pt = (right_eye.x * w, right_eye.y * h)

        dx = right_eye_pt[0] - left_eye_pt[0]
        dy = right_eye_pt[1] - left_eye_pt[1]
        angle = math.degrees(math.atan2(dy, dx))

        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(
            face_crop, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        nose = landmarks[4]
        chin = landmarks[152]

        eyes_center_x = (left_eye.x + right_eye.x) / 2.0
        eyes_center_y = (left_eye.y + right_eye.y) / 2.0
        inter_eye_dist = abs(right_eye.x - left_eye.x)

        yaw = 0.0
        if inter_eye_dist > 0.01:
            yaw_ratio = (nose.x - eyes_center_x) / (inter_eye_dist / 2.0)
            yaw = float(yaw_ratio * 30.0)

        eye_to_chin = abs(chin.y - eyes_center_y)
        pitch = 0.0
        if eye_to_chin > 0.01:
            expected_nose_y = eyes_center_y + eye_to_chin * 0.40
            pitch_ratio = (nose.y - expected_nose_y) / (eye_to_chin * 0.5)
            pitch = float(pitch_ratio * 30.0)

        return aligned, (yaw, pitch)
    except Exception as e:
        if SHOW_DEBUG:
            print(f"  [Alignment] Lỗi: {e}")
        return face_crop, None

def quality_gate(face_crop: np.ndarray) -> Tuple[bool, str, Optional[np.ndarray]]:
    if face_crop is None or face_crop.size == 0:
        return False, "empty_image", None

    h, w = face_crop.shape[:2]
    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
        return False, "too_small", None

    blur = compute_blur_score(face_crop)
    if blur < MIN_FACE_BLUR_REALTIME:
        return False, f"too_blurry ({blur:.1f})", None

    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    if mean_brightness < MIN_BRIGHTNESS:
        return False, f"too_dark ({mean_brightness:.0f})", None
    if mean_brightness > MAX_BRIGHTNESS:
        return False, f"too_bright ({mean_brightness:.0f})", None
    if contrast < MIN_CONTRAST:
        return False, f"low_contrast ({contrast:.1f})", None

    aligned, pose = align_face(face_crop)
    if aligned is None:
        return False, "alignment_failed", None

    if pose is not None:
        yaw, pitch = pose
        if abs(yaw) > MAX_YAW_ANGLE:
            return False, f"face_turned ({yaw:.0f} deg)", None
        if abs(pitch) > MAX_PITCH_ANGLE:
            return False, f"face_tilted ({pitch:.0f} deg)", None

    return True, "ok", aligned

def get_dataset_signature(items: List[Tuple[str, str]]) -> str:
    parts = []
    for person_name, image_path in sorted(items, key=lambda x: (x[0], x[1])):
        try:
            stat = os.stat(image_path)
            parts.append(f"{person_name}|{image_path}|{stat.st_size}|{stat.st_mtime}")
        except OSError:
            parts.append(f"{person_name}|{image_path}|missing")
    joined = "\n".join(parts).encode("utf-8", errors="ignore")
    return hashlib.md5(joined).hexdigest()

def person_score_from_embeddings(probe_emb: np.ndarray, ref_embeddings: List[np.ndarray]) -> float:
    if len(ref_embeddings) == 0:
        return -1.0
    sims = sorted((compute_cosine_similarity(probe_emb, ref_emb) for ref_emb in ref_embeddings), reverse=True)
    top_k = sims[: min(3, len(sims))]
    if len(top_k) == 1:
        return float(top_k[0])
    return float(0.75 * top_k[0] + 0.25 * (sum(top_k) / len(top_k)))

def get_all_face_images(dataset_dir: str) -> List[Tuple[str, str]]:
    items = []
    for entry in os.listdir(dataset_dir):
        full_path = os.path.join(dataset_dir, entry)
        if os.path.isfile(full_path) and entry.lower().endswith((".jpg", ".jpeg", ".png")):
            name = os.path.splitext(entry)[0].split(" (")[0].strip()
            items.append((name, full_path))
        elif os.path.isdir(full_path):
            person_name = entry.strip()
            for file_name in os.listdir(full_path):
                file_path = os.path.join(full_path, file_name)
                if os.path.isfile(file_path) and file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    items.append((person_name, file_path))
    return items

def prefilter_database_embeddings(person_name: str, embs: List[np.ndarray], fpaths: List[str]) -> List[np.ndarray]:
    if len(embs) < 2:
        return embs
    kept = list(range(len(embs)))
    for _ in range(2): 
        if len(kept) < 2: break
        centroid = l2_normalize(np.mean(np.stack([embs[i] for i in kept]), axis=0))
        new_kept = []
        for i in kept:
            sim = compute_cosine_similarity(embs[i], centroid)
            if sim >= INTRA_CLASS_MIN_SIM:
                new_kept.append(i)
            else:
                fname = os.path.basename(fpaths[i])
                print(f"  [Tiền lọc] Đã loại ảnh nhiễm chéo '{fname}' (Độ lệch chuẩn quá cao)")
        if not new_kept:
            new_kept = [max(kept, key=lambda i: compute_cosine_similarity(embs[i], centroid))]
        kept = new_kept
    return [embs[i] for i in kept]

def detect_single_face_box(img: np.ndarray, yolo_model) -> Optional[Tuple[int, int, int, int]]:
    try:
        results = yolo_model(img, verbose=False)
        if len(results) == 0 or results[0].boxes is None:
            return None
        boxes = results[0].boxes
        if len(boxes) != 1:
            return None
        x1, y1, x2, y2 = map(int, boxes.xyxy[0])
        return (x1, y1, x2, y2)
    except Exception:
        return None

def crop_best_face(img: np.ndarray, yolo_model) -> Optional[np.ndarray]:
    box = detect_single_face_box(img, yolo_model)
    if box is None:
        return None
    crop = crop_with_padding(img, box, FACE_PADDING_RATIO)
    if crop.size == 0:
        return None
    return crop

def build_face_embedding(face_crop: np.ndarray) -> Optional[np.ndarray]:
    if face_crop is None or face_crop.size == 0:
        return None
    enhanced_img = enhance_face_image(face_crop)
    try:
        emb_objs = DeepFace.represent(
            img_path=enhanced_img,
            model_name=EMBED_MODEL_NAME,
            enforce_detection=False,
        )
        if not emb_objs:
            return None
            
        best_emb = emb_objs[0]["embedding"]
        max_area = 0
        for obj in emb_objs:
            area = obj.get("facial_area", {})
            w = area.get("w", 0)
            h = area.get("h", 0)
            if w * h > max_area:
                max_area = w * h
                best_emb = obj["embedding"]
                
        embedding = np.array(best_emb, dtype=np.float32)
        return l2_normalize(embedding)
    except Exception:
        return None

def load_known_faces(dataset_path: str, yolo_model, cache_file: str = CACHE_FILE) -> Dict[str, List[np.ndarray]]:
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    all_items = get_all_face_images(dataset_path)
    if len(all_items) == 0:
        print(f"[!] Chưa có ảnh mẫu trong thư mục '{dataset_path}'.")
        return {}

    dataset_signature = get_dataset_signature(all_items)

    if os.path.exists(cache_file):
        try:
            print(f"[*] Tìm thấy cache '{cache_file}', đang kiểm tra...")
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            if data.get("signature") == dataset_signature and isinstance(data.get("db"), dict):
                print("[*] Cache hợp lệ, dùng lại dữ liệu khuôn mặt.")
                return data["db"]
            print("[*] Cache không còn hợp lệ, sẽ tạo lại.")
        except Exception as e:
            print(f"[!] Lỗi đọc cache: {e}. Sẽ tạo lại cache.")

    print("[*] Đang tạo cơ sở dữ liệu khuôn mặt...")
    person_to_embs: Dict[str, List[np.ndarray]] = {}

    for person_name, image_path in all_items:
        try:
            img = cv2.imread(image_path)
            if img is None:
                continue

            face_crop = crop_best_face(img, yolo_model)
            if face_crop is None:
                continue

            h, w = face_crop.shape[:2]
            if h < MIN_FACE_SIZE or w < MIN_FACE_SIZE:
                continue

            if compute_blur_score(face_crop) < 20.0:
                continue

            aligned_crop, _ = align_face(face_crop)
            if aligned_crop is not None:
                face_crop = aligned_crop

            embedding = build_face_embedding(face_crop)
            if embedding is None:
                continue

            person_to_embs.setdefault(person_name, []).append((embedding, image_path))
            print(f"  + OK: {person_name} -> {os.path.basename(image_path)}")

        except Exception as e:
            print(f"  - Bỏ qua '{image_path}': {e}")

    print("[*] Đang chạy bộ tiền lọc dữ liệu chống nhận nhầm...")
    clean_db: Dict[str, List[np.ndarray]] = {}
    for person_name in sorted(person_to_embs.keys()):
        data = person_to_embs[person_name]
        embs = [x[0] for x in data]
        paths = [x[1] for x in data]
        
        filtered_embs = prefilter_database_embeddings(person_name, embs, paths)
        if len(filtered_embs) > 0:
            clean_db[person_name] = [l2_normalize(np.asarray(emb, dtype=np.float32)) for emb in filtered_embs]

    if clean_db:
        try:
            with open(cache_file, "wb") as f:
                pickle.dump({"db": clean_db, "signature": dataset_signature}, f)
            print(f"[*] Đã lưu cache với {len(clean_db)} người.")
        except Exception as e:
            print(f"[!] Không lưu được cache: {e}")

    return clean_db

def recognize_face(face_crop: np.ndarray, face_w: int, known_face_db: dict) -> Tuple[str, float]:
    if face_crop is None or face_crop.size == 0:
        return "Loi AI", 0.0

    passed, reason, aligned_crop = quality_gate(face_crop)
    if not passed:
        if SHOW_DEBUG:
            print(f"  [QualityGate] Loại bỏ: {reason}")
        return "Khong chac", 0.0

    working_crop = aligned_crop if aligned_crop is not None else face_crop.copy()
    probe_emb = build_face_embedding(working_crop)
    
    if probe_emb is None or len(known_face_db) == 0:
        return "Unknown", 0.0

    person_scores = {}
    for person_name, ref_embeddings in known_face_db.items():
        person_scores[person_name] = person_score_from_embeddings(probe_emb, ref_embeddings)

    ranked = sorted(person_scores.items(), key=lambda x: x[1], reverse=True)
    if len(ranked) == 0:
        return "Unknown", 0.0

    best_name, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else None

    threshold = get_face_threshold(face_w)
    gap_threshold = get_gap_threshold(face_w)

    if second_score is None:
        if SHOW_DEBUG:
            print(f"  [Debug AI] Top1: {best_name} ({best_score:.3f}) | Ngưỡng: {threshold:.3f} (Không có người thứ 2)")
        if best_score > threshold + 0.05:
            return best_name, float(best_score)
        if best_score >= UNKNOWN_MIN_SIM:
            return "Khong chac", float(best_score)
        return "Loi AI", float(best_score)

    gap = float(best_score - second_score)
    
    if SHOW_DEBUG:
        second_name = ranked[1][0]
        print(f"  [Debug AI] Top1: {best_name} ({best_score:.3f}) | Top2: {second_name} ({second_score:.3f}) | Gap: {gap:.3f} | Ngưỡng: {threshold:.3f} | Ngưỡng Gap: {gap_threshold:.3f}")

    if best_score >= threshold and gap >= gap_threshold:
        return best_name, float(best_score)

    if best_score >= UNKNOWN_MIN_SIM:
        if gap < gap_threshold:
            return "Khong chac", float(best_score)
        return "Unknown", float(best_score)

    return "Loi AI", float(best_score)
