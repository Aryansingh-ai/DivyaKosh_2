"""
infer.py

Runs the trained MobileNetV3 classifier on a single image (or a live phone/
webcam frame) and applies the confidence-based ACCEPT/REVIEW decision logic.

This produces the structured output shape described in the project spec
(trimmed to what's actually implemented at this stage -- no YOLO/OCR yet,
so those fields are omitted rather than faked).

USAGE (single image):
    python src/infer.py --image path/to/note.jpg --checkpoint models/mobilenet_v3_baseline.pt

USAGE (webcam / phone as webcam, press SPACE to capture, ESC to quit):
    python src/infer.py --webcam --checkpoint models/mobilenet_v3_baseline.pt

CONFIDENCE THRESHOLD:
Default is 0.85, but this is a placeholder until you've actually looked at
the val-set confidence distribution (see evaluate_threshold.py, next step)
for correct vs incorrect predictions. Don't ship the default without
checking it against your own data.
"""

import argparse
import json
import time

import cv2
import torch
from PIL import Image
from torchvision import transforms

from train import build_model  # reuse the same architecture definition


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]
    model = build_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def classify_image(model, class_names, pil_image, device, threshold, pocket_id="webcam-0"):
    tf = get_eval_transform()
    tensor = tf(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top_prob, top_idx = torch.max(probs, dim=0)

    predicted_denomination = class_names[top_idx.item()]
    confidence = round(top_prob.item(), 4)
    decision = "ACCEPT" if confidence >= threshold else "REVIEW"

    result = {
        "pocket_id": pocket_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "item_status": "NOTE_DETECTED",  # placeholder until YOLO validity check is added
        "predicted_denomination": predicted_denomination,
        "classifier_confidence": confidence,
        "decision": decision,
        "target_bin": f"BIN_{predicted_denomination}" if decision == "ACCEPT" else "BIN_REVIEW",
        "model_version": "mobilenet_v3_baseline_v0",
    }
    return result


def run_single_image(args, model, class_names, device):
    image = Image.open(args.image).convert("RGB")
    result = classify_image(model, class_names, image, device, args.threshold)
    print(json.dumps(result, indent=2))


def run_webcam(args, model, class_names, device):
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {args.camera_index}. "
            f"If using your phone as a webcam, check it's connected/selected "
            f"as the active camera device and try a different --camera_index."
        )

    print("Press SPACE to capture and classify, ESC to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break
        cv2.imshow("Daan Drishti - press SPACE to capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            result = classify_image(model, class_names, pil_image, device, args.threshold)
            print(json.dumps(result, indent=2))

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="models/mobilenet_v3_baseline.pt")
    parser.add_argument("--image", type=str, default=None, help="Path to a single image to classify")
    parser.add_argument("--webcam", action="store_true", help="Run live webcam/phone-cam capture loop")
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.85,
                         help="Confidence threshold for ACCEPT vs REVIEW. "
                              "TUNE THIS against your val set before demoing.")
    args = parser.parse_args()

    if not args.image and not args.webcam:
        parser.error("Provide either --image <path> or --webcam")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_model(args.checkpoint, device)
    print(f"Loaded model. Classes: {class_names}")

    if args.image:
        run_single_image(args, model, class_names, device)
    else:
        run_webcam(args, model, class_names, device)


if __name__ == "__main__":
    main()
