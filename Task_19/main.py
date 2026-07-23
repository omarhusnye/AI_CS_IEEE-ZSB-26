"""
Fruit Detection System - YOLOv8 Inference Script
--------------------------------------------------
Runs a trained YOLOv8 model (best.pt) on either a live webcam feed or an
offline video file, drawing bounding boxes + labels + confidence scores,
displaying real-time FPS, and counting how many times a chosen object 
class appears in the stream.

Usage:
    Live camera:
        python main.py --mode camera --weights best.pt

    Video file:
        python main.py --mode video --weights best.pt --source input.mp4 --output result.mp4

    With object counter (e.g. count every "apple" detection):
        python main.py --mode video --weights best.pt --source input.mp4 --count-class apple
"""

import argparse
import time
from collections import defaultdict

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 fruit detection - live camera / video file inference")
    parser.add_argument("--mode", choices=["camera", "video"], required=True,
                         help="Run on a live webcam feed or an offline video file")
    parser.add_argument("--weights", type=str, default="best.pt",
                         help="Path to the trained YOLOv8 weights file")
    parser.add_argument("--source", type=str, default=None,
                         help="Path to the input video file (required when --mode video). "
                              "Ignored in camera mode.")
    parser.add_argument("--camera-id", type=int, default=0,
                         help="Webcam device index (default: 0)")
    parser.add_argument("--output", type=str, default="output_annotated.mp4",
                         help="Path to save the annotated video (video mode only)")
    parser.add_argument("--conf", type=float, default=0.4,
                         help="Confidence threshold for detections")
    parser.add_argument("--count-class", type=str, default=None,
                         help="Class name to count and print each time it appears")
    return parser.parse_args()


def draw_detections(frame, results, count_class, class_counts):
    """Draw bounding boxes + labels + confidence, and update the bonus counter."""
    boxes = results.boxes
    names = results.names

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = names[cls_id]

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label + confidence
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 2, cv2.LINE_AA)

        # Count a specific class every time it appears
        if count_class is not None and label.lower() == count_class.lower():
            class_counts[label] += 1
            print(f"[COUNT] {label} detected -> total so far: {class_counts[label]}")

    return frame


def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return frame


def run_camera(model, args):
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_id}")

    class_counts = defaultdict(int)
    prev_time = time.time()

    print("Press 'q' to quit live camera mode.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        results = model.predict(frame, conf=args.conf, verbose=False)[0]
        frame = draw_detections(frame, results, args.count_class, class_counts)

        now = time.time()
        fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
        prev_time = now
        frame = draw_fps(frame, fps)

        cv2.imshow("Fruit Detection - Live Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if args.count_class:
        print(f"\nFinal count of '{args.count_class}': {class_counts[args.count_class]}")


def run_video(model, args):
    if not args.source:
        raise ValueError("--source is required in video mode")

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, src_fps, (width, height))

    class_counts = defaultdict(int)
    prev_time = time.time()
    frame_idx = 0

    print(f"Processing {total_frames} frames from '{args.source}'...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model.predict(frame, conf=args.conf, verbose=False)[0]
        frame = draw_detections(frame, results, args.count_class, class_counts)

        now = time.time()
        fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
        prev_time = now
        frame = draw_fps(frame, fps)

        writer.write(frame)

        if frame_idx % 30 == 0:
            print(f"  processed {frame_idx}/{total_frames} frames")

    cap.release()
    writer.release()

    print(f"\nDone. Annotated video saved to: {args.output}")
    if args.count_class:
        print(f"Final count of '{args.count_class}': {class_counts[args.count_class]}")


def main():
    args = parse_args()
    model = YOLO(args.weights)

    if args.mode == "camera":
        run_camera(model, args)
    else:
        run_video(model, args)


if __name__ == "__main__":
    main()
