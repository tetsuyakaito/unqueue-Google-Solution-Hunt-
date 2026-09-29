"""
Unqueue crowd counter (hackathon demo)

- Loops a demo clip (sample.mp4) or live camera feed.
- All processing happens on-device with Ultralytics YOLOv8n.
- Class 0 (person) is detected, optionally filtered by ZONE_POLYGON, and smoothed over 5 readings.
- In LIVE mode, updates Firestore document crowd/{PLACE_ID} with {count, updatedAt, source}.
- In DRY_RUN mode, counts are printed to terminal without uploading.

Run:
  python yolo_counter.py             (Runs with settings below; default is DRY_RUN = False if credentials exist)
  python yolo_counter.py --dry-run   (Dry run only, no upload)
  python yolo_counter.py --live      (Forces live upload to Firestore)
"""
import argparse
import os
import sys
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

# ==================================================================
#  SETTINGS
# ==================================================================

# Set to True for testing YOLO locally without writing to Firebase.
# Set to False to stream counts directly to Firestore crowd/{PLACE_ID}.
DRY_RUN = False
SHOW_PREVIEW = True

# Firebase connection mode: "admin" (service account) or "rest" (REST API)
FIREBASE_MODE = "admin"

# Possible service account filenames to look for
SERVICE_ACCOUNT_FILES = [
    "serviceAccountKey.json",
    "unqueue-b7902-firebase-adminsdk-fbsvc-d472b94fe9.json",
]

# REST API settings fallback
REST_PROJECT_ID = "unqueue-b7902"
REST_API_KEY = "AIzaSyBeklzjhkPj9ZsSJ--8PO73kcQ1rVXt_BU"

VIDEO_FILE = "sample.mp4"         # Video file to loop in the same directory
PLACE_ID = "mess"                 # Target place document: crowd/mess
INTERVAL_SECONDS = 2.5            # Analyze one frame every 2.5 seconds
SMOOTHING_READINGS = 5            # Moving average over the last 5 readings
CONFIDENCE = 0.35                 # Detection threshold for person (class 0)
MODEL_FILE = "yolov8n.pt"         # Automatically downloaded on first run
PREVIEW_WIDTH = 960               # Display window width

# Optional counting polygon [(x, y), ...]. None = whole frame.
ZONE_POLYGON = None

# ==================================================================


def find_service_account():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for fname in SERVICE_ACCOUNT_FILES:
        path = os.path.join(script_dir, fname)
        if os.path.exists(path):
            return path
        if os.path.exists(fname):
            return fname
    return None


def make_sender(dry_run=DRY_RUN):
    """Returns a send(count) function or None if dry-running."""
    if dry_run:
        print("[Unqueue] DRY RUN mode active: counts are displayed locally and NOT sent to Firebase.")
        return None

    if FIREBASE_MODE == "admin":
        key_path = find_service_account()
        if not key_path:
            print(
                f"[Unqueue Warning] Service account JSON not found ({SERVICE_ACCOUNT_FILES}). "
                "Falling back to DRY RUN mode."
            )
            return None

        try:
            import firebase_admin
            from firebase_admin import credentials, firestore

            if not firebase_admin._apps:
                cred = credentials.Certificate(key_path)
                firebase_admin.initialize_app(cred)
            db = firestore.client()

            def send(count):
                db.collection("crowd").document(PLACE_ID).set(
                    {
                        "count": int(count),
                        "updatedAt": firestore.SERVER_TIMESTAMP,
                        "source": "camera",
                    }
                )

            print(f"[Unqueue] Connected to Firebase via Admin SDK ({os.path.basename(key_path)}).")
            print(f"[Unqueue] Streaming live crowd counts to Firestore: crowd/{PLACE_ID}")
            return send

        except Exception as e:
            print(f"[Unqueue Warning] Firebase Admin init failed ({e}). Check if Firestore API is enabled.")
            print("[Unqueue] Continuing in DRY RUN mode...")
            return None

    if FIREBASE_MODE == "rest":
        import requests

        base = f"projects/{REST_PROJECT_ID}/databases/(default)/documents"
        url = f"https://firestore.googleapis.com/v1/{base}:commit?key={REST_API_KEY}"

        def send_rest(count):
            body = {
                "writes": [
                    {
                        "update": {
                            "name": f"{base}/crowd/{PLACE_ID}",
                            "fields": {
                                "count": {"integerValue": str(int(count))},
                                "source": {"stringValue": "camera"},
                            },
                        },
                        "updateTransforms": [
                            {"fieldPath": "updatedAt", "setToServerValue": "REQUEST_TIME"}
                        ],
                    }
                ]
            }
            r = requests.post(url, json=body, timeout=10)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")

        print("[Unqueue] Connected to Firebase via REST API.")
        return send_rest

    return None


def detect_people(model, frame, zone):
    """Detects class 0 (person) within optional zone polygon."""
    results = model(frame, classes=[0], conf=CONFIDENCE, verbose=False)[0]
    boxes = []
    if results.boxes is not None:
        for x1, y1, x2, y2 in results.boxes.xyxy.cpu().numpy():
            if zone is not None:
                feet = (float((x1 + x2) / 2), float(y2))
                if cv2.pointPolygonTest(zone, feet, False) < 0:
                    continue
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return boxes


def main():
    parser = argparse.ArgumentParser(description="Unqueue YOLO Crowd Counter")
    parser.add_argument("--dry-run", action="store_true", help="Print counts locally, don't upload")
    parser.add_argument("--live", action="store_true", help="Force upload to Firestore")
    parser.add_argument("--video", type=str, default=VIDEO_FILE, help="Path to video file")
    parser.add_argument("--webcam", action="store_true", help="Use built-in/USB webcam instead of video file")
    parser.add_argument("--camera", type=int, default=None, help="Camera device index (e.g. 0)")
    parser.add_argument("--no-gui", action="store_true", help="Run without cv2 window (headless)")
    args = parser.parse_args()

    is_dry = DRY_RUN
    if args.dry_run:
        is_dry = True
    elif args.live:
        is_dry = False

    if args.webcam or args.camera is not None:
        cam_idx = 0 if args.camera is None else args.camera
        source_desc = f"Webcam (device {cam_idx})"
        cap = cv2.VideoCapture(cam_idx)
    else:
        video_path = args.video
        if not os.path.exists(video_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            alt_path = os.path.join(script_dir, video_path)
            if os.path.exists(alt_path):
                video_path = alt_path
            else:
                print(f"[Error] Video file '{video_path}' not found.")
                sys.exit(1)
        source_desc = video_path
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[Error] Cannot open video source: {source_desc}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1:
        fps = 25
    delay = max(1, int(1000 / fps))

    print(f"[Unqueue] Loading YOLO model ({MODEL_FILE})...")
    model = YOLO(MODEL_FILE)
    send = make_sender(dry_run=is_dry)
    zone = np.array(ZONE_POLYGON, np.int32) if ZONE_POLYGON else None

    history = deque(maxlen=SMOOTHING_READINGS)
    last_run = 0.0
    boxes, raw, smooth = [], 0, 0
    just_looped = False
    global SHOW_PREVIEW
    if args.no_gui:
        SHOW_PREVIEW = False
        
    video_writer = None
    if not SHOW_PREVIEW:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter('output_preview.mp4', fourcc, fps, (width, height))

    print("\n" + "=" * 55)
    print(" Unqueue Crowd Counter running!")
    print(f" Source: {source_desc}")
    print(f" Mode:   {'DRY RUN (Terminal only)' if not send else 'LIVE (Streaming to Firestore)'}")
    print(" Click video preview window and press 'q' to quit.")
    print("=" * 55 + "\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if just_looped:
                    print("[Error] Failed to read frames after loop restart.")
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                just_looped = True
                continue
            just_looped = False

            now = time.time()
            if now - last_run >= INTERVAL_SECONDS:
                last_run = now
                boxes = detect_people(model, frame, zone)
                raw = len(boxes)
                history.append(raw)
                smooth = int(round(sum(history) / len(history)))
                status_tag = "[LIVE UPLOAD]" if send else "[DRY RUN]"
                print(f"{status_tag} People detected: {raw:2d} | 5-sample smoothed: {smooth:2d}")

                if send:
                    try:
                        send(smooth)
                    except Exception as e:
                        print(f"[Firestore write failed]: {e}")

            # Preview UI overlay
            display_frame = frame.copy()
            if zone is not None:
                cv2.polylines(display_frame, [zone], True, (255, 200, 0), 2)
            for x1, y1, x2, y2 in boxes:
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            tag = "DRY RUN" if not send else "LIVE FIRESTORE"
            cv2.putText(
                display_frame,
                f"Crowd: {smooth} (Raw: {raw}) [{tag}]",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )

            if SHOW_PREVIEW:
                try:
                    if display_frame.shape[1] > PREVIEW_WIDTH:
                        scale = PREVIEW_WIDTH / display_frame.shape[1]
                        disp_resized = cv2.resize(
                            display_frame,
                            (PREVIEW_WIDTH, int(display_frame.shape[0] * scale)),
                        )
                    else:
                        disp_resized = display_frame

                    cv2.imshow("Unqueue - YOLO Crowd Counter", disp_resized)
                    key = cv2.waitKey(delay) & 0xFF
                    if key == ord("q") or key == 27:
                        break
                except cv2.error:
                    SHOW_PREVIEW = False
                    print("[Unqueue] GUI display not available; switching to video writer fallback.")
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter('output_preview.mp4', fourcc, fps, (width, height))
                    video_writer.write(display_frame)
            else:
                if video_writer is not None:
                    video_writer.write(display_frame)
                time.sleep(delay / 1000.0)

    except KeyboardInterrupt:
        print("\nStopping crowd counter...")
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if SHOW_PREVIEW:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        print("[Unqueue] Exited cleanly.")


if __name__ == "__main__":
    main()
