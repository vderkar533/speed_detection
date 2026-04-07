# Speed Detection

This project detects vehicles in a camera stream, estimates their speed inside a selected ROI, and stores violation snapshots in a log folder.

## Files

- `main.py` runs vehicle detection, tracking, speed estimation, and violation logging.
- `logs/` is created automatically when the app runs.
- `models/` is the suggested place to keep YOLO model weights locally.

## Setup

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set environment variables before running:

```powershell
$env:VIDEO_PATH="rtsp://username:password@camera-ip:554/stream1"
$env:MODEL_PATH="models/high_mast_model_vehicle_detection.pt"
python main.py
```

## Notes

- Do not upload private RTSP credentials to GitHub.
- Large `.pt` model files are ignored by `.gitignore`.
- Generated logs and snapshots are also ignored.
