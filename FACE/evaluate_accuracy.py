import os
import cv2
import time
from ultralytics import YOLO

# Import các hàm và cấu hình có sẵn của hệ thống
from config import YOLO_MODEL_PATH, KNOWN_FACES_DIR, YOLO_CONFIDENCE, FACE_PADDING_RATIO
from face_processing import load_known_faces, recognize_face
from utils import crop_with_padding

TEST_DIR = "test_faces"

def run_evaluation():
    # Kiểm tra xem thư mục test_faces đã tồn tại chưa
    if not os.path.exists(TEST_DIR):
        print(f"\n[!] Thư mục chứa ảnh test '{TEST_DIR}' chưa tồn tại.")
        print(f"[*] Hệ thống đã tự động tạo thư mục '{TEST_DIR}'.")
        print(f"[*] HƯỚNG DẪN: Bạn hãy copy ảnh cần test vào đây và chia theo từng người.")
        print(f"    Cấu trúc thư mục ví dụ:")
        print(f"    {TEST_DIR}/")
        print(f"    ├── LeGiaHuy/      (Chứa các ảnh test của Huy - Không trùng với ảnh trong known_faces)")
        print(f"    │   ├── test1.jpg")
        print(f"    │   └── test2.jpg")
        print(f"    ├── NguyenVanA/")
        print(f"    │   └── test1.jpg")
        print(f"    └── Unknown/       (BẮT BUỘC NÊN CÓ: Chứa ảnh người lạ để test hệ thống có nhận nhầm không)")
        print(f"        └── la1.jpg")
        os.makedirs(TEST_DIR, exist_ok=True)
        os.makedirs(os.path.join(TEST_DIR, "Unknown"), exist_ok=True)
        return

    print("\n[*] Đang khởi tạo mô hình YOLO...")
    try:
        model = YOLO(YOLO_MODEL_PATH)
    except Exception as e:
        print(f"[!] Không thể nạp mô hình YOLO: {e}")
        return

    print(f"[*] Đang nạp Cơ sở dữ liệu khuôn mặt mẫu từ '{KNOWN_FACES_DIR}'...")
    known_face_db = load_known_faces(KNOWN_FACES_DIR, model)
    if not known_face_db:
        print("[!] Không có dữ liệu người quen. Hãy kiểm tra thư mục known_faces.")
        return

    # Các biến thống kê (Metrics)
    total_images = 0
    correct_predictions = 0
    false_acceptances = 0 # Tỷ lệ FAR: Người lạ bị nhận nhầm thành người quen
    false_rejections = 0  # Tỷ lệ FRR: Người quen bị nhận thành Unknown hoặc nhận sai tên người khác
    no_face_errors = 0
    
    y_true = []
    y_pred = []
    
    print("\n" + "="*60)
    print("BẮT ĐẦU CHẠY ĐÁNH GIÁ ĐỘ CHÍNH XÁC (EVALUATION BENCHMARK)")
    print("="*60)
    
    start_time = time.time()

    # Quét qua từng thư mục con (tương ứng với tên từng người) trong test_faces
    for person_name in os.listdir(TEST_DIR):
        person_dir = os.path.join(TEST_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
            
        for img_name in os.listdir(person_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                continue
                
            img_path = os.path.join(person_dir, img_name)
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"[!] Không thể đọc ảnh: {img_path}")
                continue
                
            total_images += 1
            true_label = person_name # Nhãn thực tế lấy từ tên thư mục
            
            # 1. Phát hiện khuôn mặt bằng YOLO
            results = model.predict(frame, conf=YOLO_CONFIDENCE, verbose=False)
            boxes = results[0].boxes
            
            score_str = "N/A"
            if boxes is None or len(boxes) == 0:
                pred_label = "NoFace"
                no_face_errors += 1
            else:
                # 2. Lấy khuôn mặt lớn nhất trong ảnh để test
                largest_box = None
                max_area = 0
                for box in boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        largest_box = (x1, y1, x2, y2)
                
                x1, y1, x2, y2 = largest_box
                face_w = x2 - x1
                face_crop = crop_with_padding(frame, largest_box, FACE_PADDING_RATIO)
                
                if face_crop.size == 0:
                    pred_label = "CropError"
                    no_face_errors += 1
                else:
                    # 3. Trích xuất đặc trưng và Nhận diện (ArcFace)
                    pred_name, score = recognize_face(face_crop, face_w, known_face_db)
                    pred_label = pred_name
                    score_str = f"{score:.2f}"
                    
            y_true.append(true_label)
            y_pred.append(pred_label)
            
            # 4. Đối chiếu kết quả dự đoán với thực tế
            is_correct = (true_label == pred_label)
            
            # Nếu người test là Unknown (người lạ), hệ thống trả về "Unknown", "Khong chac" hoặc "Loi AI" đều được coi là ĐÚNG (Vì nó đã ngăn không cho người lạ điểm danh)
            if true_label == "Unknown" and pred_label in ["Unknown", "Khong chac", "Loi AI"]:
                is_correct = True
            
            # Ghi nhận chi tiết các lỗi
            status = "✅ ĐÚNG" if is_correct else "❌ SAI "
            if not is_correct:
                if true_label == "Unknown" and pred_label not in ["Unknown", "Khong chac", "Loi AI", "NoFace", "CropError"]:
                    status += f"(False Acceptance: Nhận nhầm người LẠ thành {pred_label})"
                    false_acceptances += 1
                elif true_label != "Unknown" and pred_label in ["Unknown", "Khong chac", "Loi AI"]:
                    status += f"(False Rejection: Không nhận ra {true_label}, AI báo: {pred_label})"
                    false_rejections += 1
                elif true_label != "Unknown" and pred_label not in ["Unknown", "Khong chac", "Loi AI", "NoFace", "CropError"]:
                    status += f"(Nhận nhầm người này thành {pred_label})"
                    false_rejections += 1
            else:
                correct_predictions += 1
                
            print(f"[{person_name}/{img_name}] | Thực tế: {true_label:12} | Dự đoán: {pred_label:12} | Điểm: {score_str} -> {status}")
            
    end_time = time.time()
    
    print("\n" + "="*60)
    print("BÁO CÁO TỔNG KẾT HIỆU SUẤT ĐỘ CHÍNH XÁC (METRICS)")
    print("="*60)
    
    if total_images == 0:
        print("[!] Không tìm thấy bức ảnh nào để test. Hãy thêm ảnh vào thư mục test_faces.")
        return
        
    accuracy = (correct_predictions / total_images) * 100
    
    print(f"1. Tổng số ảnh kiểm thử: {total_images} ảnh")
    print(f"2. Thời gian chạy Test:  {end_time - start_time:.2f} giây")
    print(f"   => Tốc độ trung bình:   {(end_time - start_time) / total_images:.3f} giây / ảnh")
    print("-" * 60)
    print(f"3. Số lần nhận diện ĐÚNG: {correct_predictions}")
    print(f"4. Số lần nhận diện SAI:  {total_images - correct_predictions}")
    print(f"   Trong đó chi tiết lỗi:")
    print(f"   • False Acceptances (Nhận Lạ -> Quen): {false_acceptances} lần (Rất nguy hiểm, cần = 0)")
    print(f"   • False Rejections (Quen -> Lạ / Nhầm): {false_rejections} lần")
    print(f"   • Không tìm thấy khuôn mặt trong ảnh:  {no_face_errors} lần")
    print("-" * 60)
    print(f"🎯 ĐỘ CHÍNH XÁC TỔNG THỂ (ACCURACY): {accuracy:.2f}%")
    print("="*60)
    
    print("\n[MẸO BÁO CÁO]:")
    print(" - Hãy copy bảng tổng kết này (vào Word/Excel) để làm báo cáo minh chứng độ chính xác.")
    print(" - Thời gian nhận diện mỗi ảnh (Tốc độ trung bình) chính là độ trễ (Latency) xử lý AI của hệ thống.\n")

if __name__ == "__main__":
    run_evaluation()
