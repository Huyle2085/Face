from dataclasses import dataclass, field
from typing import Tuple, Optional, List
import numpy as np

from config import (
    INVALID_NAMES,
    MAX_WEAK_CONSECUTIVE,
    REQUIRED_CONFIRMATIONS,
    ALLOW_ONE_WEAK_FRAME,
    EVIDENCE_PADDING_RATIO
)
from utils import get_face_threshold, crop_with_padding
# We will import compute_face_quality locally in the function to avoid circular imports if any,
# or we can import it at the top if face_processing doesn't import tracker.
from face_processing import compute_face_quality

@dataclass
class Detection:
    box: Tuple[int, int, int, int]
    confidence: float = 0.0
    name: str = "Unknown"
    score: float = 0.0
    need_recognition: bool = True

@dataclass
class TrackState:
    box: Tuple[int, int, int, int]
    missing: int = 0
    best_name: str = "Unknown"
    best_score: float = 0.0
    confirmed: bool = False
    attendance_marked: bool = False

    streak_name: Optional[str] = None
    streak_count: int = 0
    weak_consecutive: int = 0  # Đếm số frame yếu liên tiếp
    score_history: List[float] = field(default_factory=list)

    snapshot: Optional[np.ndarray] = None
    snapshot_quality: float = -1.0
    last_recognized_frame: int = -99999

def update_track_identity(track: TrackState, name: str, score: float):
    if name in INVALID_NAMES:
        if track.streak_name is not None:
            track.weak_consecutive += 1
            if track.weak_consecutive <= MAX_WEAK_CONSECUTIVE:
                # Cho phép vài frame yếu liên tiếp mà không reset streak
                return
            else:
                # Quá nhiều frame yếu liên tiếp → reset
                track.streak_name = None
                track.streak_count = 0
                track.weak_consecutive = 0
                track.score_history.clear()
        return

    # Reset weak frame counter khi có frame tốt
    track.weak_consecutive = 0

    if track.streak_name == name:
        track.streak_count += 1
    else:
        track.streak_name = name
        track.streak_count = 1
        track.score_history = []

    track.score_history.append(score)
    track.score_history = track.score_history[-REQUIRED_CONFIRMATIONS:]

    avg_score = float(np.mean(track.score_history))
    if avg_score > track.best_score:
        track.best_score = avg_score
        track.best_name = name

    face_w = track.box[2] - track.box[0]
    threshold = get_face_threshold(face_w)

    if track.streak_count >= REQUIRED_CONFIRMATIONS:
        if ALLOW_ONE_WEAK_FRAME:
            weak_frames = sum(s < threshold for s in track.score_history)
            if weak_frames <= 1 and avg_score >= threshold:
                track.confirmed = True
                track.best_name = name
                track.best_score = avg_score
        else:
            all_scores_valid = all(s >= threshold for s in track.score_history)
            if all_scores_valid and avg_score >= threshold:
                track.confirmed = True
                track.best_name = name
                track.best_score = avg_score

def maybe_update_snapshot(track: TrackState, frame: np.ndarray, box, recog_score: float):
    evidence = crop_with_padding(frame, box, EVIDENCE_PADDING_RATIO)
    if evidence.size == 0:
        return

    quality = compute_face_quality(evidence, recog_score)
    if quality > track.snapshot_quality:
        track.snapshot = evidence
        track.snapshot_quality = quality
