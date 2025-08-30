# src/splitter.py
import cv2
import numpy as np
import os
from scipy.signal import find_peaks
import math

def make_even(x: int) -> int:
    return x if x % 2 == 0 else x-1

def split_video(video_path, train_number, output_root,
                     sample_step=10, min_coach_px=80,
                     edge_weight=0.6, motion_weight=0.4,
                     smooth_k=21, valley_k=51):
    """
    Detect coach vertical boundaries combining edge profile and motion profile.
    Save spatial-cropped per-coach videos under output_root/<train_number>_<i>/
    Returns list of coach metadata dicts.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fps = int(round(fps))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    height = make_even(height)

    # 1) sample frames to compute edge profile and motion profile
    edge_profiles = []
    motion_profiles = []
    prev_gray = None
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_profiles.append(edges.sum(axis=0).astype(np.float32))
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                motion_profiles.append(diff.sum(axis=0).astype(np.float32))
            prev_gray = gray
        idx += 1
    cap.release()

    if len(edge_profiles) == 0:
        # fallback: equal split into 4 parts
        approx = max(1, width // 250)
        seg_w = width // approx
        regions = [(i*seg_w, (i+1)*seg_w if i<approx-1 else width) for i in range(approx)]
    else:
        # average and normalize
        edge_avg = np.mean(edge_profiles, axis=0)
        edge_s = np.convolve(edge_avg, np.ones(smooth_k)/smooth_k, mode='same')
        if len(motion_profiles) > 0:
            motion_avg = np.mean(motion_profiles, axis=0)
            motion_s = np.convolve(motion_avg, np.ones(smooth_k)/smooth_k, mode='same')
        else:
            motion_s = np.zeros_like(edge_s)

        # normalize to [0,1]
        def norm(v):
            v = v - v.min()
            if v.max() > 0:
                return v / v.max()
            return v
        e_n = norm(edge_s)
        m_n = norm(motion_s)
        combined = edge_weight * e_n + motion_weight * m_n

        # invert to find valleys between coaches
        inv = 1.0 - combined
        inv_smooth = np.convolve(inv, np.ones(valley_k)/valley_k, mode='same')

        # find valleys (peaks in inv_smooth)
        min_distance = max(10, min_coach_px//2)
        peaks, props = find_peaks(inv_smooth, height=0.30, distance=min_distance)
        gap_positions = peaks.tolist()

        boundaries = [0] + gap_positions + [width-1]
        boundaries = sorted(list(set(boundaries)))
        regions = []
        for i in range(len(boundaries)-1):
            x0 = int(boundaries[i])
            x1 = int(boundaries[i+1])
            if (x1 - x0) >= min_coach_px:
                regions.append((x0, x1))

        if len(regions) == 0:
            approx = max(1, width // 250)
            seg_w = width // approx
            regions = [(i*seg_w, (i+1)*seg_w if i<approx-1 else width) for i in range(approx)]

    # create output folders and writers
    out_base = os.path.join(output_root, train_number)
    os.makedirs(out_base, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writers = []
    coach_meta = []
    for i, (x0, x1) in enumerate(regions, start=1):
        w = x1 - x0
        w = make_even(w)
        if w <= 0:
            continue
        h = height
        coach_folder = os.path.join(out_base, f"{train_number}_{i}")
        os.makedirs(coach_folder, exist_ok=True)
        out_path = os.path.join(coach_folder, f"{train_number}_{i}.mp4")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        writers.append((writer, x0, w, out_path, coach_folder))
        coach_meta.append({
            "coach_index": i,
            "folder": coach_folder,
            "video": out_path,
            "x0": x0,
            "x1": x1
        })

    # write frames (single pass)
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[0:h, 0:make_even(width)]
        for (writer, x0, w, out_path, coach_folder) in writers:
            crop = frame[:, x0:x0+w]
            if crop.shape[1] != w or crop.shape[0] != h:
                crop = cv2.resize(crop, (w, h))
            writer.write(crop)
    cap.release()
    for (writer, _, _, _, _) in writers:
        writer.release()

    print(f"[splitter] Detected {len(coach_meta)} coach regions for {train_number}")
    return coach_meta