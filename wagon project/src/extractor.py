import os

import cv2


def extract_frames(coach_videos, train_number, num_frames=3):
    """
    Extract representative frames (front, middle, rear) from each coach video.
    Saves frames in the same coach folder.
    
    Args:
        coach_videos (list): List of coach video paths
        train_number (str): Train number for naming
        num_frames (int): How many frames per coach (default=3 -> front, middle, rear)

    Returns:
        list: Paths of extracted frame images
    """
    extracted = []

    for vid_path in coach_videos:
        folder = os.path.dirname(vid_path)
        coach_id = os.path.basename(folder).split("_")[-1]

        cap = cv2.VideoCapture(vid_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            print(f"⚠️ Skipping {vid_path}, no frames detected")
            continue

        # Pick evenly spaced frames (front, middle, rear)
        frame_indices = [int(total_frames * i / (num_frames + 1)) for i in range(1, num_frames + 1)]

        for idx, frame_num in enumerate(frame_indices, start=1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue

            # Updated filename format: <train_number>_<counter>_<frame_number>.jpg
            img_name = f"{train_number}_{coach_id}_{idx}.jpg"
            img_path = os.path.join(folder, img_name)
            cv2.imwrite(img_path, frame)
            extracted.append(img_path)

        cap.release()

    return extracted