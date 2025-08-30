import os

from src.splitter import split_video
from src.extractor import extract_frames
from src.detector import run_yolo_on_images
from src.reportgen import build_pdf_report, build_html_report

# Hardcoded videos
VIDEOS = [
    {"path": "train1.mp4", "train_number": "TRAIN001"},
    {"path": "train2.mp4", "train_number": "TRAIN002"},
]

# Optional: YOLO weights path (set to None if not using)
WEIGHTS_PATH = None

# Output directory
OUTPUT_DIR = os.path.join(os.getcwd(), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_video_file(video_path, train_number, output_root, weights_path=None):
    print(f"\n[main] Processing {train_number} -> {video_path}")
    
    # Step 1: Split into per-coach videos
    coach_meta = split_video(video_path, train_number, output_root)
    print(f"[main] Total coaches detected: {len(coach_meta)}")

    # Step 2: Extract representative frames
    coach_videos = [m["video"] for m in coach_meta]
    extracted_frames = extract_frames(coach_videos, train_number, num_frames=3)
    print(f"[main] Extracted {len(extracted_frames)} representative frames")

    # Step 3: Run YOLO detection (or fallback)
    annotated = run_yolo_on_images(coach_meta, weights_path=weights_path, device="cpu")
    print(f"[main] Annotated {len(annotated)} frames with detections")

    # Step 4: Generate reports
    pdf_report = build_pdf_report(train_number, coach_meta, output_root)
    html_report = build_html_report(train_number, coach_meta, output_root)
    print(f"[main] Reports generated:\n  PDF: {pdf_report}\n  HTML: {html_report}")


if __name__ == "__main__":
    for vid in VIDEOS:
        video_path = vid["path"]
        train_number = vid["train_number"]

        if not os.path.exists(video_path):
            print(f"⚠️  Video not found: {video_path}, skipping...")
            continue

        process_video_file(video_path, train_number, OUTPUT_DIR, WEIGHTS_PATH)