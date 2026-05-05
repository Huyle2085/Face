from datetime import datetime

# ============================================================
# CẤU HÌNH HỆ THỐNG
# ============================================================
CAMERA_INDEX = 0
YOLO_MODEL_PATH = "yolov8n-face.pt"
FACE_LANDMARKER_MODEL = "face_landmarker.task"
KNOWN_FACES_DIR = "known_faces"
CACHE_FILE = "face_cache_arcface_v1.pkl"  # v1: Thêm Face Alignment
EVIDENCE_DIR = "attendance_evidence"
EXCEL_FILENAME = f"DanhSach_DiemDanh_{datetime.now().strftime('%d-%m-%Y')}.xlsx"

WINDOW_NAME = "He Thong Diem Danh Tu Dong"
SHOW_DEBUG = True
USE_ENTRANCE_ROI = True

# Camera / Detector
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
YOLO_CONFIDENCE = 0.45
MIN_FACE_SIZE = 36
ROI_DETECT_ZOOM = 1.6

# Recognition
EMBED_MODEL_NAME = "ArcFace"
FACE_PADDING_RATIO = 0.28
EVIDENCE_PADDING_RATIO = 0.20
RECOGNIZE_EVERY_N_FRAMES = 2
MATCH_THRESHOLD_NEAR = 0.58  # Ngưỡng tối thiểu khi mặt gần/rõ
MATCH_THRESHOLD_FAR = 0.50  # Ngưỡng khi mặt nhỏ/xa
GAP_THRESHOLD_NEAR = 0.10   # Giảm xuống để nhận diện được anh em có nét giống nhau
GAP_THRESHOLD_FAR = 0.08    # Giảm xuống tương ứng
UNKNOWN_MIN_SIM = 0.45      # Dưới mức này → Loi AI

# Tracking / Confirmation
TRACK_MAX_MISSING = 10
REQUIRED_CONFIRMATIONS = 5
ALLOW_ONE_WEAK_FRAME = True
MAX_WEAK_CONSECUTIVE = 1  # Giảm xuống 1: khắp khe hơn với frame yếu liên tiếp

INVALID_NAMES = {"Unknown", "Qua xa", "Loi AI", "Khong chac"}

# Cấu hình tiền lọc
INTRA_CLASS_MIN_SIM = 0.42     # Giảm xuống: giữ lại ảnh góc khác nhau của cùng 1 người (quan trọng để nhận đa dạng)
MIN_FACE_BLUR_REALTIME = 25.0  # Tiền lọc real-time: độ nét tối thiểu cho ảnh camera
MAX_YAW_ANGLE = 42.0           # Quality Gate: Nới lỏng - khính mắt làm lệch điểm landmark ~5-10°
MAX_PITCH_ANGLE = 30.0         # Quality Gate: Góc cúi/ngửa tối đa
MIN_BRIGHTNESS = 35            # Quality Gate: Độ sáng trung bình tối thiểu (0-255)
MAX_BRIGHTNESS = 230           # Quality Gate: Độ sáng trung bình tối đa (0-255)
MIN_CONTRAST = 15              # Quality Gate: Độ tương phản tối thiểu (std deviation)
