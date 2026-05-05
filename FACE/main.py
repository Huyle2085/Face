import cv2
import time
import threading
import traceback
from typing import Dict

from ultralytics import YOLO

from config import (
    CAMERA_INDEX, YOLO_MODEL_PATH, KNOWN_FACES_DIR, EVIDENCE_DIR,
    WINDOW_NAME, SHOW_DEBUG, USE_ENTRANCE_ROI, CAMERA_WIDTH,
    CAMERA_HEIGHT, YOLO_CONFIDENCE, MIN_FACE_SIZE, ROI_DETECT_ZOOM,
    FACE_PADDING_RATIO, RECOGNIZE_EVERY_N_FRAMES, INVALID_NAMES,
    TRACK_MAX_MISSING, REQUIRED_CONFIRMATIONS
)
from utils import (
    ensure_dir, point_in_box, make_label, get_entrance_roi, crop_with_padding
)
from excel_utils import (
    init_excel_file, load_logged_names_from_excel, mark_attendance
)
from tracker import (
    Detection, TrackState, update_track_identity, maybe_update_snapshot
)
from face_processing import (
    load_known_faces, recognize_face
)

# ============================================================
# BIẾN TOÀN CỤC
# ============================================================
latest_frame = None
latest_display = []
is_running = True
frame_lock = threading.Lock()
result_lock = threading.Lock()

ai_processing = False
ai_fps = 0.0

logged_names = set()
tracks: Dict[int, TrackState] = {}
known_face_db = {}
model = None

# ============================================================
# AI WORKER
# ============================================================
def ai_worker():
    global latest_frame, latest_display, ai_processing, ai_fps, is_running

    frame_index = 0

    while is_running:
        if ai_processing:
            time.sleep(0.005)
            continue

        with frame_lock:
            frame_to_process = None if latest_frame is None else latest_frame.copy()

        if frame_to_process is None:
            time.sleep(0.01)
            continue

        ai_processing = True
        start_t = time.time()
        frame_index += 1

        try:
            h, w = frame_to_process.shape[:2]
            entrance_roi = get_entrance_roi(w, h)
            track_to_det: Dict[int, Detection] = {}
            active_track_ids: set = set()

            try:
                if USE_ENTRANCE_ROI:
                    rx1, ry1, rx2, ry2 = entrance_roi
                    detect_region = frame_to_process[ry1:ry2, rx1:rx2].copy()
                    offset_x, offset_y = rx1, ry1
                else:
                    detect_region = frame_to_process.copy()
                    offset_x, offset_y = 0, 0

                detect_region_big = cv2.resize(
                    detect_region,
                    None,
                    fx=ROI_DETECT_ZOOM,
                    fy=ROI_DETECT_ZOOM,
                    interpolation=cv2.INTER_CUBIC,
                )

                # ---- ByteTrack: detect + track tích hợp ----
                results = model.track(
                    detect_region_big,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=YOLO_CONFIDENCE,
                    verbose=False,
                )

                for r in results:
                    if r.boxes is None or r.boxes.id is None:
                        continue

                    for i, box in enumerate(r.boxes.xyxy):
                        conf = float(r.boxes.conf[i]) if getattr(r.boxes, "conf", None) is not None else 1.0
                        track_id = int(r.boxes.id[i])

                        bx1, by1, bx2, by2 = map(int, box)
                        x1 = offset_x + int(bx1 / ROI_DETECT_ZOOM)
                        y1 = offset_y + int(by1 / ROI_DETECT_ZOOM)
                        x2 = offset_x + int(bx2 / ROI_DETECT_ZOOM)
                        y2 = offset_y + int(by2 / ROI_DETECT_ZOOM)

                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(w, x2)
                        y2 = min(h, y2)

                        face_w_det = x2 - x1
                        face_h_det = y2 - y1
                        if face_w_det <= 0 or face_h_det <= 0:
                            continue

                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        if USE_ENTRANCE_ROI and not point_in_box(cx, cy, entrance_roi):
                            continue

                        active_track_ids.add(track_id)

                        # Tạo mới hoặc cập nhật TrackState
                        if track_id not in tracks:
                            tracks[track_id] = TrackState(box=(x1, y1, x2, y2))
                        else:
                            tracks[track_id].box = (x1, y1, x2, y2)
                            tracks[track_id].missing = 0

                        if face_w_det < MIN_FACE_SIZE or face_h_det < MIN_FACE_SIZE:
                            det = Detection(box=(x1, y1, x2, y2), confidence=conf,
                                            name="Qua xa", score=0.0, need_recognition=False)
                        else:
                            det = Detection(box=(x1, y1, x2, y2), confidence=conf)

                        track_to_det[track_id] = det

            except Exception as e:
                print(f"[!] Lỗi detect/track: {e}")
                traceback.print_exc()

            # Dọn dẹp track cũ mà ByteTrack không còn theo dõi
            stale_ids = [tid for tid in tracks if tid not in active_track_ids]
            for tid in stale_ids:
                tracks[tid].missing += 1
                if tracks[tid].missing > TRACK_MAX_MISSING:
                    del tracks[tid]

            for tid, det in track_to_det.items():
                x1, y1, x2, y2 = det.box
                face_w = x2 - x1
                track = tracks[tid]

                if not det.need_recognition:
                    track.best_name = det.name
                    track.best_score = det.score
                    continue

                if track.attendance_marked:
                    continue

                if frame_index - track.last_recognized_frame < RECOGNIZE_EVERY_N_FRAMES:
                    continue

                face_crop = crop_with_padding(frame_to_process, det.box, FACE_PADDING_RATIO)
                if face_crop.size == 0:
                    continue

                track.last_recognized_frame = frame_index
                name, score = recognize_face(face_crop, face_w, known_face_db)
                update_track_identity(track, name, score)

                if SHOW_DEBUG and name not in INVALID_NAMES:
                    print(f"  [Track #{tid}] {name} score={score:.2f} streak={track.streak_count}/{REQUIRED_CONFIRMATIONS} confirmed={track.confirmed}")

                if name not in INVALID_NAMES:
                    maybe_update_snapshot(track, frame_to_process, det.box, score)

                if track.confirmed and not track.attendance_marked:
                    confirmed_name = track.best_name
                    if confirmed_name not in logged_names:
                        evidence = track.snapshot if track.snapshot is not None else face_crop
                        success = mark_attendance(confirmed_name, logged_names, evidence)
                        if success:
                            track.attendance_marked = True

            display_items = []
            for tid, tdata in tracks.items():
                x1, y1, x2, y2 = tdata.box
                show_name = tdata.best_name
                show_score = tdata.best_score

                if tdata.confirmed:
                    color = (0, 200, 0)
                elif show_name in INVALID_NAMES:
                    color = (0, 0, 255)
                else:
                    color = (0, 165, 255)

                display_items.append(
                    {
                        "track_id": tid,
                        "box": (x1, y1, x2, y2),
                        "name": show_name,
                        "score": show_score,
                        "color": color,
                        "confirmed": tdata.confirmed,
                    }
                )

            with result_lock:
                latest_display = display_items

            elapsed = time.time() - start_t
            ai_fps = 1.0 / elapsed if elapsed > 0 else 0.0

        except Exception as e:
            print(f"[!] Lỗi AI worker: {e}")
            traceback.print_exc()

        finally:
            ai_processing = False


# ============================================================
# MAIN
# ============================================================
def main():
    global model, latest_frame, is_running, known_face_db

    print("\n" + "=" * 60)
    print("HE THONG DIEM DANH TU DONG BANG NHAN DIEN KHUON MAT")
    print("Phien toi uu: giam nhan nham, on dinh hon, it goi AI thua")
    print("=" * 60)

    ensure_dir(KNOWN_FACES_DIR)
    ensure_dir(EVIDENCE_DIR)
    init_excel_file()
    load_logged_names_from_excel(logged_names)

    print("[*] Đang nạp YOLO face detector...")
    try:
        model = YOLO(YOLO_MODEL_PATH)
    except Exception as e:
        print(f"[!] Không thể nạp model YOLO: {e}")
        return

    known_face_db = load_known_faces(KNOWN_FACES_DIR, model)
    if len(known_face_db) == 0:
        print("[!] Chưa có dữ liệu khuôn mặt mẫu hợp lệ trong 'known_faces'.")
        return

    total_templates = sum(len(v) for v in known_face_db.values())
    print(f"[*] Đã nạp {len(known_face_db)} người với {total_templates} ảnh mẫu hợp lệ.")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print("[!] Không mở được webcam.")
        return

    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"[*] Camera actual resolution: {actual_w} x {actual_h}")

    ai_thread = threading.Thread(target=ai_worker, daemon=True)
    ai_thread.start()

    prev_t = time.time()

    print("\n[*] Hệ thống sẵn sàng. Nhấn ESC để thoát.")
    print("[*] Nếu ROI chưa đúng vị trí cửa lớp, hãy chỉnh get_entrance_roi().\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Mất kết nối camera.")
            break

        fh, fw = frame.shape[:2]
        entrance_roi = get_entrance_roi(fw, fh)

        with frame_lock:
            latest_frame = frame.copy()

        curr_t = time.time()
        cam_fps = 1.0 / max(1e-6, (curr_t - prev_t))
        prev_t = curr_t

        if USE_ENTRANCE_ROI:
            rx1, ry1, rx2, ry2 = entrance_roi
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 200, 0), 2)
            cv2.putText(
                frame,
                "Vung diem danh",
                (rx1, max(20, ry1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 200, 0),
                2,
            )

        with result_lock:
            faces_to_draw = latest_display.copy()

        for item in faces_to_draw:
            x1, y1, x2, y2 = item["box"]
            name = item["name"]
            score = item["score"]
            color = item["color"]
            confirmed = item["confirmed"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = make_label(name, score)
            if confirmed:
                label += " | OK"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            y_top = max(0, y1 - th - 8)
            cv2.rectangle(frame, (x1, y_top), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

        cv2.putText(frame, f"Cam FPS: {cam_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        cv2.putText(frame, f"AI FPS: {ai_fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
        cv2.putText(frame, f"So nguoi da diem danh: {len(logged_names)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        if SHOW_DEBUG:
            cv2.putText(frame, f"Tracks: {len(tracks)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (180, 255, 180), 2)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            is_running = False
            break

    cap.release()
    cv2.destroyAllWindows()
    ai_thread.join(timeout=1.0)
    print("\n[!] Đã tắt hệ thống.")


if __name__ == "__main__":
    main()
