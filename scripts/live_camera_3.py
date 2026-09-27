"""
DaanDrishti live webcam inference demo.

Flow:
    webcam -> fixed ROI -> empty-background calibration
           -> note-presence gate (OpenCV only, NOT the classifier)
           -> persistent presence confirmation (several consecutive frames)
           -> once confirmed, collect several consecutive frames and run
              MobileNetV3 on each (temporal multi-frame classification)
           -> average the softmax probability vectors across those frames
           -> ACCEPT only if the aggregated top probability >= 0.80 AND it
              leads the runner-up class by a minimum margin
           -> otherwise show LOW CONFIDENCE / UNCERTAIN, keep camera open,
              and collect a fresh batch of frames until the note is
              removed or an accepted result is reached

This adds temporal stability on top of the existing presence gate: a
single noisy frame can no longer flip the decision between denominations
(e.g. Rs.500 momentarily reading as Rs.20/Rs.50). It does NOT change the
model, the checkpoint, or the preprocessing, and it does NOT contain any
denomination-specific correction — the margin/averaging logic is generic.

Run:
    python scripts/live_camera.py
    python scripts/live_camera.py --camera 1
    python scripts/live_camera.py --checkpoint models/mobilenet_v3_clean.pt
    python scripts/live_camera.py --log_path logs/inference.jsonl

Press Q at any time before acceptance to quit without logging anything.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ============================================================
# WINDOW SETTINGS
# ============================================================

WINDOW_NAME = "DaanDrishti - Live Demo"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
WINDOW_POS_X = 50
WINDOW_POS_Y = 50


# ============================================================
# MODEL / PREPROCESSING
# ============================================================

# Preprocessing must remain EXACTLY what the model was trained with.
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

DENOM_LABELS = {
    "10": "Rs.10",
    "20": "Rs.20",
    "50": "Rs.50",
    "100": "Rs.100",
    "200": "Rs.200",
    "500": "Rs.500",
}

MODEL_VERSION = "mobilenet_v3_clean_v1"


# ============================================================
# PRESENCE GATE / TIMING CONSTANTS
# (starting values — tune later, not sacred)
# ============================================================

# Number of empty-ROI frames used to build the background model.
WARMUP_FRAMES = 20

# Pixel-difference threshold for foreground detection.
PRESENCE_PIXEL_THRESHOLD = 30

# At least this fraction of ROI pixels must differ from the background.
PRESENCE_RATIO_THRESHOLD = 0.08

# The largest connected foreground blob must occupy at least this
# fraction of the ROI (rejects small noise blobs).
MIN_CONTOUR_RATIO = 0.04

# Reject a foreground blob that fills almost the entire ROI (likely a
# lighting change or the camera being blocked, not a note).
MAX_CONTOUR_RATIO = 0.85

# Require presence for this many consecutive frames before classifying.
REQUIRED_PRESENCE_FRAMES = 4

# Confidence required to ACCEPT a prediction. Do not lower this.
ACCEPT_CONFIDENCE = 0.80

# Minimum time (seconds) between individual classification frames once a
# note is confirmed present. Prevents classifying every single frame and
# gives frames slight temporal separation (autofocus/exposure settling).
CLASSIFICATION_COOLDOWN = 0.30

# ------------------------------------------------------------------
# Temporal multi-frame classification (NEW)
# ------------------------------------------------------------------

# Number of consecutive classification frames collected (while the note
# is confirmed present) before the softmax probabilities are averaged
# and a final decision is made.
CLASSIFICATION_FRAMES = 5

# Aggregated top-1 probability must lead the runner-up class by at least
# this much. Guards against confident-but-ambiguous aggregates (e.g. a
# note that is genuinely hard to tell apart under current lighting).
MIN_CLASS_MARGIN = 0.15

# How long (seconds) to keep an UNCERTAIN/LOW-CONFIDENCE verdict on
# screen before starting to collect the next batch of frames. Purely a
# readability pause; does not affect the accept/reject logic.
VERDICT_HOLD_SECONDS = 1.0

# Debug ROI dump (temporary diagnostic feature — off by default).
DEBUG_ROI_DIR = Path("debug/live_roi")
DEBUG_MAX_FRAMES_PER_ATTEMPT = 5


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    class_names = ckpt["class_names"]

    model = models.mobilenet_v3_large(weights=None)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(class_names))

    model.load_state_dict(ckpt["model_state_dict"])

    model.to(device)
    model.eval()

    print(f"Device       : {device}")
    print(f"Model path   : {checkpoint_path}")
    print(f"Ckpt epoch   : {ckpt.get('epoch', '?')}")
    print(f"Ckpt val_acc : {ckpt.get('val_acc', '?')}")
    print(f"Classes      : {class_names}")

    return model, class_names


def save_debug_roi(attempt_idx, frame_idx, raw_roi_bgr, resized_224_rgb):
    """
    Diagnostic only. Save the exact ROI crop (pre-model, pre-normalize)
    and the exact 224x224 tensor-input image the model receives.
    Does not touch model, dataset, or inference.jsonl.
    """
    DEBUG_ROI_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = DEBUG_ROI_DIR / f"attempt_{attempt_idx:03d}_frame_{frame_idx:02d}_raw.jpg"
    resized_path = DEBUG_ROI_DIR / f"attempt_{attempt_idx:03d}_frame_{frame_idx:02d}_224.jpg"

    cv2.imwrite(str(raw_path), raw_roi_bgr)

    resized_bgr = cv2.cvtColor(resized_224_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(resized_path), resized_bgr)


# ============================================================
# CLASSIFICATION
# ============================================================

@torch.inference_mode()
def classify_roi_probs(model, class_names, roi_bgr, device,
                        debug=False, attempt_idx=0, frame_idx=0):
    """
    Run MobileNetV3 on a single ROI frame and return the FULL softmax
    probability vector (aligned with class_names), not just the top-1.
    This lets the caller aggregate probabilities across several frames
    instead of trusting one frame's argmax.

    If debug=True, saves the raw ROI and the exact 224x224 image that
    is fed to the model (post-Resize, pre-Normalize/ToTensor) so the
    actual model input can be visually inspected.
    """
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(roi_rgb)

    if debug:
        resized_pil = pil_img.resize((224, 224))
        save_debug_roi(attempt_idx, frame_idx, roi_bgr, np.array(resized_pil))

    tensor = EVAL_TRANSFORM(pil_img).unsqueeze(0).to(device)

    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0]

    return probs.cpu().numpy()


def top2(class_names, probs):
    """Return (label1, prob1, label2, prob2) sorted descending by prob."""
    order = np.argsort(probs)[::-1]
    idx1, idx2 = order[0], order[1]
    return (
        class_names[idx1], float(probs[idx1]),
        class_names[idx2], float(probs[idx2]),
    )


def format_top3(class_names, probs):
    """Terminal-friendly 'label: prob' lines for the top 3 classes."""
    order = np.argsort(probs)[::-1][:3]
    lines = []
    for idx in order:
        label = DENOM_LABELS.get(class_names[idx], class_names[idx])
        lines.append(f"{label}: {probs[idx]:.2f}")
    return lines


# ============================================================
# ROI
# ============================================================

def compute_roi(frame_h, frame_w):
    # Central region approximately matching banknote proportions.
    roi_w = int(frame_w * 0.55)
    roi_h = int(roi_w * 0.45)
    roi_h = min(roi_h, int(frame_h * 0.65))

    x1 = (frame_w - roi_w) // 2
    y1 = (frame_h - roi_h) // 2
    x2 = x1 + roi_w
    y2 = y1 + roi_h

    return x1, y1, x2, y2


# ============================================================
# IMAGE PREPROCESSING FOR PRESENCE GATE
# ============================================================

def preprocess_gray(roi_bgr):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    return gray


# ============================================================
# BACKGROUND CREATION
# ============================================================

def build_background(background_frames):
    # Median across frames is far more stable than trusting one frame.
    background = np.median(np.stack(background_frames, axis=0), axis=0)
    return background.astype(np.uint8)


# ============================================================
# NOTE PRESENCE DETECTION (OpenCV/NumPy only — never the classifier)
# ============================================================

def detect_note_presence(background_gray, current_gray):
    diff = cv2.absdiff(background_gray, current_gray)

    _, mask = cv2.threshold(
        diff, PRESENCE_PIXEL_THRESHOLD, 255, cv2.THRESH_BINARY
    )

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    roi_area = mask.shape[0] * mask.shape[1]
    foreground_pixels = cv2.countNonZero(mask)
    foreground_ratio = foreground_pixels / float(roi_area)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    largest_area = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area

    contour_ratio = largest_area / float(roi_area)

    presence_detected = (
        foreground_ratio >= PRESENCE_RATIO_THRESHOLD
        and MIN_CONTOUR_RATIO <= contour_ratio <= MAX_CONTOUR_RATIO
    )

    return presence_detected, foreground_ratio, contour_ratio


# ============================================================
# UI
# ============================================================

def draw_overlay(frame, roi_coords, label, confidence, fps, status, presence_ratio,
                  margin=None):
    x1, y1, x2, y2 = roi_coords
    h, w = frame.shape[:2]

    if status.startswith("WAITING"):
        color = (180, 180, 180)
    elif status.startswith("CHECKING") or status.startswith("CLASSIFYING"):
        color = (0, 180, 255)
    elif status.startswith("LOW CONFIDENCE"):
        color = (0, 120, 255)
    elif status.startswith("UNCERTAIN"):
        color = (0, 90, 220)
    elif status.startswith("DETECTED") or status.startswith("ACCEPT"):
        color = (0, 220, 0)
    else:
        color = (0, 180, 255)

    # ROI rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Corner brackets
    bracket = 20
    thickness = 3
    corners = [
        (x1, y1, 1, 1),
        (x2, y1, -1, 1),
        (x1, y2, 1, -1),
        (x2, y2, -1, -1),
    ]
    for cx, cy, dx, dy in corners:
        cv2.line(frame, (cx, cy), (cx + dx * bracket, cy), color, thickness)
        cv2.line(frame, (cx, cy), (cx, cy + dy * bracket), color, thickness)

    # Instruction above ROI
    cv2.putText(
        frame, "Place note inside the box", (x1, max(25, y1 - 12)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA,
    )

    if label is None:
        cv2.putText(
            frame, status, (x1, y1 + (y2 - y1) // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA,
        )
    else:
        display_label = DENOM_LABELS.get(label, label)
        confidence_text = f"{confidence * 100:.1f}%"
        if margin is not None:
            confidence_text += f"   margin {margin * 100:.1f}%"

        panel_y = min(h - 65, y2 + 35)

        cv2.putText(
            frame, f"{display_label}  {confidence_text}", (x1, panel_y),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA,
        )

        # Status line (e.g. LOW CONFIDENCE — HOLD NOTE STEADY) under the label.
        status_y = min(h - 45, panel_y + 25)
        cv2.putText(
            frame, status, (x1, status_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA,
        )

        # Confidence bar
        bar_x = x1
        bar_y = min(h - 25, status_y + 12)
        bar_w = x2 - x1
        bar_h = 8

        cv2.rectangle(
            frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
            (80, 80, 80), -1,
        )

        fill_width = int(bar_w * min(confidence, 1.0))
        cv2.rectangle(
            frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_h),
            color, -1,
        )

    # FPS (top-right)
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (max(10, w - 140), 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA,
    )

    # Title (top-left)
    cv2.putText(
        frame, "DaanDrishti", (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA,
    )

    # Presence diagnostic (bottom-left)
    cv2.putText(
        frame, f"Presence: {presence_ratio * 100:.1f}%", (15, h - 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA,
    )

    cv2.putText(
        frame, "Q = quit", (15, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA,
    )


# ============================================================
# LOGGING
# ============================================================

def append_log(log_path, camera_index, label, confidence):
    """
    Called exactly once per successful script execution, only when
    confidence >= ACCEPT_CONFIDENCE. Always writes decision = ACCEPT.
    Appends to the file; never truncates or rewrites existing lines.
    """

    record = {
        "pocket_id": f"webcam-{camera_index}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "item_status": "NOTE_DETECTED",
        "predicted_denomination": label,
        "classifier_confidence": confidence,
        "confidence": confidence,
        "source": "webcam",
        "decision": "ACCEPT",
        "target_bin": f"BIN_{label}",
        "model_version": MODEL_VERSION,
    }

    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")

    return record


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="DaanDrishti live camera note detection + classification"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="models/mobilenet_v3_clean.pt"
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--log_path", type=str, default="logs/inference.jsonl"
    )
    parser.add_argument(
        "--debug_roi", action="store_true",
        help="Save raw + 224x224 model-input ROI frames to debug/live_roi/ "
             "for diagnosing live-vs-training domain mismatch. Diagnostic "
             "only — does not touch model, dataset, or inference.jsonl.",
    )
    args = parser.parse_args()

    # ------------------------------------------
    # Device + model
    # ------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_model(args.checkpoint, device)

    # ------------------------------------------
    # Camera
    # ------------------------------------------

    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print(
            f"ERROR: Could not open camera index {args.camera}. "
            f"Try --camera 1."
        )
        return

    log_path = Path(args.log_path)

    # ------------------------------------------
    # Window — created and sized BEFORE the loop so it is
    # reliably visible on Windows.
    # ------------------------------------------

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.moveWindow(WINDOW_NAME, WINDOW_POS_X, WINDOW_POS_Y)

    print()
    print(f"Camera opened (index {args.camera}).")
    print("KEEP THE ROI EMPTY during startup calibration.")
    print("Waiting for a real note...")
    if args.debug_roi:
        print(f"DEBUG_ROI enabled. Saving frames to: {DEBUG_ROI_DIR}")
    print()

    # ------------------------------------------
    # State
    # ------------------------------------------

    background_frames = []
    background_gray = None

    presence_streak = 0
    note_confirmed = False
    note_confirmed_announced = False

    # Per-frame preview (most recent single-frame classification, shown
    # live while a batch is being collected).
    last_label = None
    last_confidence = 0.0
    last_classification_time = 0.0

    # Temporal aggregation state (NEW).
    collected_probs = []          # list of full probability vectors
    last_verdict_label = None     # aggregated result of the last full batch
    last_verdict_confidence = 0.0
    last_verdict_margin = 0.0
    last_verdict_status = "LOW CONFIDENCE - HOLD NOTE STEADY"
    verdict_hold_until = 0.0      # while now < this, don't collect; just display
    attempt_idx = 0                # increments once per full batch (debug naming)

    frame_counter = 0
    prev_time = time.perf_counter()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Failed to read frame.")
                break

            # ROI
            h, w = frame.shape[:2]
            roi_coords = compute_roi(h, w)
            x1, y1, x2, y2 = roi_coords
            roi_crop = frame[y1:y2, x1:x2]
            roi_gray = preprocess_gray(roi_crop)

            # FPS
            now = time.perf_counter()
            dt = now - prev_time
            if dt > 0:
                fps = 1.0 / dt
            prev_time = now

            # ==================================================
            # BACKGROUND CALIBRATION
            # ==================================================

            if background_gray is None:
                background_frames.append(roi_gray.copy())
                frame_counter += 1

                status = f"CALIBRATING BACKGROUND {frame_counter}/{WARMUP_FRAMES}"

                if frame_counter >= WARMUP_FRAMES:
                    background_gray = build_background(background_frames)
                    print("Background calibration complete.")
                    status = "WAITING FOR NOTE"

                draw_overlay(frame, roi_coords, None, 0.0, fps, status, 0.0)
                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                continue

            # ==================================================
            # PRESENCE DETECTION (every frame — lightweight)
            # ==================================================

            presence_detected, presence_ratio, contour_ratio = detect_note_presence(
                background_gray, roi_gray
            )

            if presence_detected:
                presence_streak += 1
            else:
                presence_streak = 0

                # If a note had been confirmed and is now gone, reset
                # completely and go back to waiting. Never mix frames
                # from different physical objects into one aggregate.
                if note_confirmed:
                    note_confirmed = False
                    note_confirmed_announced = False
                    last_label = None
                    last_confidence = 0.0
                    collected_probs = []
                    last_verdict_label = None
                    last_verdict_confidence = 0.0
                    last_verdict_margin = 0.0
                    last_verdict_status = "LOW CONFIDENCE - HOLD NOTE STEADY"
                    verdict_hold_until = 0.0

            # ==================================================
            # CONFIRM PRESENCE (persistent, not single-frame)
            # ==================================================

            if not note_confirmed and presence_streak >= REQUIRED_PRESENCE_FRAMES:
                note_confirmed = True
                last_classification_time = 0.0  # force immediate classification
                attempt_idx += 1

                if args.debug_roi:
                    roi_w = x2 - x1
                    roi_h = y2 - y1
                    print()
                    print(f"--- Attempt {attempt_idx:03d} ---")
                    print(f"Frame dimensions : {w} x {h}")
                    print(f"ROI coordinates  : x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                    print(f"ROI width        : {roi_w}")
                    print(f"ROI height       : {roi_h}")
                    print(f"Raw ROI dims     : {roi_crop.shape[1]} x {roi_crop.shape[0]}")
                    print("Resized to 224x224 before model : YES")

            # ==================================================
            # CLASSIFY ONLY AFTER PRESENCE IS CONFIRMED
            # ==================================================

            if note_confirmed:
                if not note_confirmed_announced:
                    print("Physical object detected in ROI.")
                    note_confirmed_announced = True

                time_since_last = now - last_classification_time

                holding_verdict = now < verdict_hold_until

                # ------------------------------------------------------
                # Collect one more classification frame into the batch,
                # unless we're pausing to let the last verdict be read.
                # ------------------------------------------------------
                if not holding_verdict and time_since_last >= CLASSIFICATION_COOLDOWN:
                    probs = classify_roi_probs(
                        model, class_names, roi_crop, device,
                        debug=args.debug_roi,
                        attempt_idx=attempt_idx,
                        frame_idx=len(collected_probs) + 1,
                    )
                    last_classification_time = now

                    collected_probs.append(probs)

                    frame_label, frame_conf, _, _ = top2(class_names, probs)
                    last_label, last_confidence = frame_label, frame_conf

                    top3_lines = format_top3(class_names, probs)
                    print(
                        f"CHECKING... {len(collected_probs)}/{CLASSIFICATION_FRAMES}  "
                        + "  ".join(top3_lines)
                    )

                    # --------------------------------------------------
                    # Batch complete -> aggregate and decide.
                    # --------------------------------------------------
                    if len(collected_probs) >= CLASSIFICATION_FRAMES:
                        aggregated = np.mean(np.stack(collected_probs, axis=0), axis=0)
                        top_label, top_prob, second_label, second_prob = top2(
                            class_names, aggregated
                        )
                        margin = top_prob - second_prob

                        last_verdict_label = top_label
                        last_verdict_confidence = top_prob
                        last_verdict_margin = margin

                        collected_probs = []  # ready for next batch either way

                        if top_prob >= ACCEPT_CONFIDENCE and margin >= MIN_CLASS_MARGIN:
                            # ------------------------------------------
                            # ACCEPT
                            # ------------------------------------------
                            target_bin = f"BIN_{top_label}"

                            print()
                            print("FINAL:")
                            print(
                                f"  {DENOM_LABELS.get(top_label, top_label)} "
                                f"{top_prob * 100:.1f}%   Margin: {margin:.2f}"
                            )

                            draw_overlay(
                                frame, roi_coords, top_label, top_prob,
                                fps, "ACCEPTED", presence_ratio, margin=margin,
                            )
                            cv2.imshow(WINDOW_NAME, frame)
                            cv2.waitKey(300)  # keep result visible briefly

                            record = append_log(
                                log_path, args.camera, top_label, top_prob
                            )

                            print()
                            print("=== FINAL RECORD ===")
                            print(json.dumps(record, indent=2))
                            print()
                            print(f"One record appended to: {log_path}")
                            print(f"Target bin: {target_bin}")

                            # Exactly one detection per execution.
                            break

                        else:
                            # ------------------------------------------
                            # REJECT this batch — decide why, for the UI,
                            # then hold the verdict on screen briefly
                            # before collecting a fresh batch.
                            # ------------------------------------------
                            if top_prob < ACCEPT_CONFIDENCE:
                                last_verdict_status = "LOW CONFIDENCE - HOLD NOTE STEADY"
                                print(
                                    f"LOW CONFIDENCE  Top: "
                                    f"{DENOM_LABELS.get(top_label, top_label)} "
                                    f"{top_prob * 100:.1f}%   Margin: {margin:.2f}"
                                )
                            else:
                                last_verdict_status = "UNCERTAIN - HOLD NOTE STEADY"
                                print(
                                    f"UNCERTAIN  Top: "
                                    f"{DENOM_LABELS.get(top_label, top_label)} "
                                    f"{top_prob * 100:.1f}%   Margin: {margin:.2f}"
                                )

                            verdict_hold_until = now + VERDICT_HOLD_SECONDS

                # ------------------------------------------------------
                # Draw current state: either mid-batch progress, or the
                # held verdict from the batch that just finished.
                # ------------------------------------------------------
                if holding_verdict or verdict_hold_until > now:
                    status = last_verdict_status
                    display_label = last_verdict_label
                    display_conf = last_verdict_confidence
                    display_margin = last_verdict_margin
                else:
                    status = f"CLASSIFYING {len(collected_probs)}/{CLASSIFICATION_FRAMES}"
                    display_label = last_label
                    display_conf = last_confidence
                    display_margin = None

                draw_overlay(
                    frame, roi_coords, display_label, display_conf,
                    fps, status, presence_ratio, margin=display_margin,
                )
                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                continue

            # ==================================================
            # WAITING / CHECKING UI (no object confirmed yet)
            # ==================================================

            if presence_detected:
                status = f"CHECKING... {presence_streak}/{REQUIRED_PRESENCE_FRAMES}"
            else:
                status = "WAITING FOR NOTE"

            draw_overlay(frame, roi_coords, None, 0.0, fps, status, presence_ratio)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released. Goodbye.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()