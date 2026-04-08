"""
Script debug để kiểm tra vấn đề nhận diện
"""
import cv2
import os
import numpy as np
from deepface import DeepFace
from ultralytics import YOLO

# ============================================================
# CẤU HÌNH GIỐNG main.py
# ============================================================
YOLO_MODEL_PATH = "yolov8n-face.pt"
KNOWN_FACES_DIR = "known_faces"
EMBED_MODEL_NAME = "ArcFace"
FACE_PADDING_RATIO = 0.28
MIN_FACE_SIZE = 55
MATCH_THRESHOLD_NEAR = 0.65
MATCH_THRESHOLD_FAR = 0.57

# ============================================================
# HÀM PHỤ
# ============================================================
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

def detect_single_face_box(img: np.ndarray, yolo_model):
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

def build_face_embedding(face_crop: np.ndarray):
    try:
        emb_objs = DeepFace.represent(
            img_path=face_crop,
            model_name=EMBED_MODEL_NAME,
            enforce_detection=False,
        )
        if len(emb_objs) == 0:
            return None
        embedding = np.array(emb_objs[0]["embedding"], dtype=np.float32)
        return l2_normalize(embedding)
    except Exception as e:
        print(f"  ❌ Lỗi tạo embedding: {e}")
        return None

# ============================================================
# KIỂM TRA DỮ LIỆU MẪU
# ============================================================
def check_known_faces():
    print("\n" + "="*60)
    print("📊 KIỂM TRA DỮ LIỆU KHUÔN MẶT")
    print("="*60)
    
    model = YOLO(YOLO_MODEL_PATH)
    
    all_items = []
    for entry in os.listdir(KNOWN_FACES_DIR):
        full_path = os.path.join(KNOWN_FACES_DIR, entry)
        if os.path.isfile(full_path) and entry.lower().endswith((".jpg", ".jpeg", ".png")):
            name = os.path.splitext(entry)[0].split(" (")[0].strip()
            all_items.append((name, full_path))
    
    person_counts = {}
    for person, _ in all_items:
        person_counts[person] = person_counts.get(person, 0) + 1
    
    print(f"\n👤 Tổng số người: {len(person_counts)}")
    for person in sorted(person_counts.keys()):
        print(f"  • {person}: {person_counts[person]} ảnh")
    
    # Kiểm tra từng ảnh
    print(f"\n🔍 Chi tiết từng ảnh:")
    failed_images = []
    embeddings_per_person = {}
    
    for person_name, image_path in all_items:
        img = cv2.imread(image_path)
        if img is None:
            print(f"  ❌ {image_path}: Không đọc được ảnh")
            failed_images.append((person_name, image_path, "Không đọc được file"))
            continue
        
        # Kiểm tra size ảnh
        h, w = img.shape[:2]
        
        # Detect faces
        box = detect_single_face_box(img, model)
        if box is None:
            print(f"  ❌ {os.path.basename(image_path)}: Không detect được khuôn mặt")
            failed_images.append((person_name, image_path, "Không detect được khuôn mặt"))
            continue
        
        x1, y1, x2, y2 = box
        face_w = x2 - x1
        face_h = y2 - y1
        
        if face_w < MIN_FACE_SIZE or face_h < MIN_FACE_SIZE:
            print(f"  ⚠️  {os.path.basename(image_path)}: Khuôn mặt quá nhỏ ({face_w}x{face_h}px, tối thiểu {MIN_FACE_SIZE}x{MIN_FACE_SIZE}px)")
            failed_images.append((person_name, image_path, f"Khuôn mặt quá nhỏ ({face_w}x{face_h}px)"))
            continue
        
        # Crop face và tạo embedding
        face_crop = crop_with_padding(img, box, FACE_PADDING_RATIO)
        if face_crop.size == 0:
            print(f"  ❌ {os.path.basename(image_path)}: Lỗi crop ảnh")
            failed_images.append((person_name, image_path, "Lỗi crop ảnh"))
            continue
        
        embed = build_face_embedding(face_crop)
        if embed is None:
            print(f"  ❌ {os.path.basename(image_path)}: Lỗi tạo embedding")
            failed_images.append((person_name, image_path, "Lỗi tạo embedding"))
            continue
        
        print(f"  ✅ {os.path.basename(image_path)}: OK (size {face_w}x{face_h}px, embedding OK)")
        
        if person_name not in embeddings_per_person:
            embeddings_per_person[person_name] = []
        embeddings_per_person[person_name].append(embed)
    
    # Tổng kết các lỗi
    if failed_images:
        print(f"\n⚠️  TỔNG SỐ ẢNH LỖI: {len(failed_images)}")
        for person, path, reason in failed_images:
            print(f"  • {person}: {os.path.basename(path)} - {reason}")
    
    # Kiểm tra variance embedding
    print(f"\n📈 Phân tích embedding per person:")
    for person, embeddings in embeddings_per_person.items():
        if len(embeddings) > 1:
            # Tính độ tương tự giữa các embedding của cùng một người
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    sim = compute_cosine_similarity(embeddings[i], embeddings[j])
                    similarities.append(sim)
            
            avg_sim = np.mean(similarities)
            min_sim = np.min(similarities)
            max_sim = np.max(similarities)
            print(f"  {person}: {len(embeddings)} ảnh")
            print(f"    - Tương tự trung bình: {avg_sim:.4f}")
            print(f"    - Min-Max: {min_sim:.4f} - {max_sim:.4f}")

# ============================================================
# TEST NHẬN DIỆN CAMERA
# ============================================================
def test_camera_recognition():
    print("\n" + "="*60)
    print("📷 TEST NHẬN DIỆN CAMERA")
    print("="*60)
    print("\nKhai báo webcam... (nhấn 'q' để thoát, 'c' để chup ảnh test)")
    
    model = YOLO(YOLO_MODEL_PATH)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Không thể mở webcam!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Load known faces
    all_items = []
    for entry in os.listdir(KNOWN_FACES_DIR):
        full_path = os.path.join(KNOWN_FACES_DIR, entry)
        if os.path.isfile(full_path) and entry.lower().endswith((".jpg", ".jpeg", ".png")):
            name = os.path.splitext(entry)[0].split(" (")[0].strip()
            all_items.append((name, full_path))
    
    known_face_encodings = []
    known_face_names = []
    person_to_embs = {}
    
    for person_name, image_path in all_items:
        img = cv2.imread(image_path)
        if img is None:
            continue
        box = detect_single_face_box(img, model)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        if (x2-x1) < MIN_FACE_SIZE or (y2-y1) < MIN_FACE_SIZE:
            continue
        face_crop = crop_with_padding(img, box, FACE_PADDING_RATIO)
        embed = build_face_embedding(face_crop)
        if embed is None:
            continue
        person_to_embs.setdefault(person_name, []).append(embed)
    
    for person_name in sorted(person_to_embs.keys()):
        embs = person_to_embs[person_name]
        mean_emb = np.mean(np.stack(embs, axis=0), axis=0)
        mean_emb = l2_normalize(mean_emb)
        known_face_encodings.append(mean_emb.astype(np.float32))
        known_face_names.append(person_name)
    
    print(f"\n✅ Đã load {len(known_face_names)} người: {', '.join(known_face_names)}\n")
    
    frame_count = 0
    test_faces = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (1280, 720))
        frame_count += 1
        
        # Detect
        results = model(frame, verbose=False)
        faces_in_frame = []
        
        for r in results:
            if r.boxes is None:
                continue
            for i, box in enumerate(r.boxes.xyxy):
                conf = float(r.boxes.conf[i]) if hasattr(r.boxes, "conf") else 1.0
                x1, y1, x2, y2 = map(int, box)
                face_w = x2 - x1
                face_h = y2 - y1
                
                if face_w < MIN_FACE_SIZE or face_h < MIN_FACE_SIZE:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "TOO SMALL", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                    continue
                
                face_crop = crop_with_padding(frame, (x1, y1, x2, y2), FACE_PADDING_RATIO)
                embed = build_face_embedding(face_crop)
                
                if embed is None:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, "EMBED FAIL", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
                    continue
                
                # Compare
                similarities = [compute_cosine_similarity(embed, known_emb) for known_emb in known_face_encodings]
                best_idx = np.argmax(similarities)
                best_name = known_face_names[best_idx]
                best_sim = similarities[best_idx]
                
                threshold = MATCH_THRESHOLD_NEAR if face_w > 80 else MATCH_THRESHOLD_FAR
                
                color = (0, 255, 0) if best_sim > threshold else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"{best_name}: {best_sim:.3f} (threshold: {threshold:.2f})"
                cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                
                faces_in_frame.append({
                    "name": best_name,
                    "sim": best_sim,
                    "threshold": threshold,
                    "match": best_sim > threshold
                })
        
        cv2.putText(frame, f"Frame: {frame_count} | Faces: {len(faces_in_frame)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("Debug Recognition", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Lưu frame để test
            test_filename = f"test_face_{frame_count}.jpg"
            cv2.imwrite(test_filename, frame)
            print(f"\n💾 Đã lưu test frame: {test_filename}")
            if faces_in_frame:
                print(f"   Khuôn mặt trong frame:")
                for face_info in faces_in_frame:
                    match_status = "✅ MATCH" if face_info["match"] else "❌ NO MATCH"
                    print(f"     • {face_info['name']}: {face_info['sim']:.4f} (threshold: {face_info['threshold']:.2f}) {match_status}")
            else:
                print(f"   Không detect được khuôn mặt nào!")
    
    cap.release()
    cv2.destroyAllWindows()

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n🔧 HỆ THỐNG DEBUG NHẬN DIỆN KHUÔN MẶT\n")
    
    # Bước 1: Kiểm tra dữ liệu mẫu
    check_known_faces()
    
    # Bước 2: Test camera
    print("\n\n")
    input("Nhấn ENTER để bắt đầu test camera...")
    test_camera_recognition()
    
    print("\n✅ Debug hoàn thành!")
