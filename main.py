import csv
import os
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|"
    "fflags;nobuffer|"
    "flags;low_delay|"
    "max_delay;0|"
    "probesize;32|"
    "analyzeduration;0"
)

cv2.setNumThreads(1)


MODEL_PATH = os.getenv("MODEL_PATH", "models/high_mast_model_vehicle_detection.pt")
VIDEO_PATH = os.getenv("VIDEO_PATH", "")

PREVIEW_W, PREVIEW_H = 1280, 720
CONF = 0.4
IOU = 0.5
SPEED_LIMIT_KMH = 15.0
SMOOTHING_WINDOW = 5

ROI_POINTS = [
    (868, 715),
    (1063, 715),
    (813, 411),
    (735, 413),
]

ROI_WIDTH_M = 8.7
ROI_HEIGHT_M = 86.6

WARP_W = 600
WARP_H = int(WARP_W * ROI_HEIGHT_M / ROI_WIDTH_M)
PIXELS_PER_METER = WARP_W / ROI_WIDTH_M

SNAPSHOT_LINE_Y1 = int(WARP_H * 0.30)
SNAPSHOT_LINE_Y2 = int(WARP_H * 0.70)

LOG_DIR = "logs"
SNAPSHOT_DIR = os.path.join(LOG_DIR, "snapshots")
LOG_CSV_PATH = os.path.join(LOG_DIR, "speed_violations.csv")

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

if not os.path.exists(LOG_CSV_PATH):
    with open(LOG_CSV_PATH, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["track_id", "speed_kmh", "timestamp", "snapshot_path"])


src = np.array(ROI_POINTS, dtype=np.float32)
dst = np.array(
    [
        [0, 0],
        [WARP_W, 0],
        [WARP_W, WARP_H],
        [0, WARP_H],
    ],
    dtype=np.float32,
)
warp_matrix = cv2.getPerspectiveTransform(src, dst)


def point_inside_roi(point, roi):
    return cv2.pointPolygonTest(np.array(roi), point, False) >= 0


def point_to_warp(point):
    pt = np.array([[[point[0], point[1]]]], dtype=np.float32)
    warp_pt = cv2.perspectiveTransform(pt, warp_matrix)[0][0]
    return int(warp_pt[0]), int(warp_pt[1])


def point_between_snapshot_lines(warp_y):
    low = min(SNAPSHOT_LINE_Y1, SNAPSHOT_LINE_Y2)
    high = max(SNAPSHOT_LINE_Y1, SNAPSHOT_LINE_Y2)
    return low <= warp_y <= high


def draw_snapshot_lines(warped_frame):
    line_color = (0, 255, 255)
    cv2.line(warped_frame, (0, SNAPSHOT_LINE_Y1), (WARP_W, SNAPSHOT_LINE_Y1), line_color, 2)
    cv2.line(warped_frame, (0, SNAPSHOT_LINE_Y2), (WARP_W, SNAPSHOT_LINE_Y2), line_color, 2)
    cv2.putText(
        warped_frame,
        "Capture Zone",
        (10, min(SNAPSHOT_LINE_Y1, SNAPSHOT_LINE_Y2) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        line_color,
        2,
    )


def save_violation(track_id, speed_kmh, frame, box):
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    vehicle_crop = frame[y1:y2, x1:x2]
    if vehicle_crop.size == 0:
        return None

    timestamp = datetime.now()
    timestamp_text = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    filename = f"id_{track_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
    snapshot_path = os.path.join(SNAPSHOT_DIR, filename)
    cv2.imwrite(snapshot_path, vehicle_crop)

    with open(LOG_CSV_PATH, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([track_id, f"{speed_kmh:.2f}", timestamp_text, snapshot_path])

    return snapshot_path


def main():
    if not VIDEO_PATH:
        raise ValueError("Set VIDEO_PATH before running the project.")

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow("YOLO Track", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO Track", PREVIEW_W, PREVIEW_H)
    cv2.namedWindow("Warped ROI", cv2.WINDOW_NORMAL)

    tracks = {}

    while True:
        for _ in range(2):
            cap.grab()

        ret, frame = cap.read()
        if not ret:
            continue

        now = time.time()

        results = model.track(
            frame,
            conf=CONF,
            iou=IOU,
            persist=True,
            tracker="bytetrack.yaml",
        )

        cv2.polylines(frame, [np.array(ROI_POINTS)], True, (255, 0, 0), 2)

        warped = cv2.warpPerspective(frame, warp_matrix, (WARP_W, WARP_H))
        draw_snapshot_lines(warped)

        for r in results:
            if r.boxes is None or r.boxes.id is None:
                continue

            for box, track_id in zip(r.boxes.xyxy, r.boxes.id):
                x1, y1, x2, y2 = map(int, box)
                tid = int(track_id)

                cx = int((x1 + x2) / 2)
                cy = int(y2)
                cam_point = (cx, cy)
                inside = point_inside_roi(cam_point, ROI_POINTS)

                wx, wy = point_to_warp(cam_point)
                warp_point = (wx, wy)
                in_snapshot_zone = inside and point_between_snapshot_lines(wy)

                if tid not in tracks:
                    tracks[tid] = {
                        "inside": False,
                        "last_point": None,
                        "last_time": None,
                        "speed_kmh": 0.0,
                        "speed_history": deque(maxlen=SMOOTHING_WINDOW),
                        "violation_logged": False,
                        "snapshot_taken": False,
                        "snapshot_path": None,
                    }

                t = tracks[tid]

                if inside and not t["inside"]:
                    t["inside"] = True
                    t["last_point"] = warp_point
                    t["last_time"] = now
                    t["speed_history"].clear()
                    t["speed_kmh"] = 0.0
                    t["violation_logged"] = False
                    t["snapshot_taken"] = False
                    t["snapshot_path"] = None
                elif inside and t["inside"]:
                    px, py = t["last_point"]
                    dt = now - t["last_time"]

                    if dt > 0:
                        pixel_dist = np.hypot(wx - px, wy - py)
                        meters = pixel_dist / PIXELS_PER_METER
                        speed_kmh = (meters / dt) * 3.6
                        t["speed_history"].append(speed_kmh)
                        t["speed_kmh"] = float(np.mean(t["speed_history"]))

                    t["last_point"] = warp_point
                    t["last_time"] = now
                elif not inside and t["inside"]:
                    t["inside"] = False

                is_violating = inside and t["speed_kmh"] > SPEED_LIMIT_KMH

                if in_snapshot_zone and not t["snapshot_taken"]:
                    t["snapshot_taken"] = True

                if is_violating and in_snapshot_zone and not t["violation_logged"]:
                    t["snapshot_path"] = save_violation(tid, t["speed_kmh"], frame, (x1, y1, x2, y2))
                    t["violation_logged"] = t["snapshot_path"] is not None

                color = (0, 0, 255) if is_violating else ((0, 255, 0) if inside else (0, 180, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, cam_point, 4, (0, 0, 255), -1)

                label = f"ID {tid}"
                if inside:
                    label += f" | {t['speed_kmh']:.1f} km/h"
                if is_violating:
                    label += " | VIOLATION"

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                if 0 <= wx < WARP_W and 0 <= wy < WARP_H:
                    cv2.circle(warped, (wx, wy), 4, (0, 255, 0), -1)

        preview = cv2.resize(frame, (PREVIEW_W, PREVIEW_H))
        cv2.imshow("YOLO Track", preview)
        cv2.imshow("Warped ROI", warped)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
