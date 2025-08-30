# src/detector.py
import os
import cv2
from tqdm import tqdm

# Try to use ultralytics YOLO model if available
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except Exception:
    HAS_ULTRALYTICS = False

def _annotate_and_save(img_path, detections, backup_original=True):
    img = cv2.imread(img_path)
    if img is None:
        return None
    if backup_original:
        bak = img_path + ".bak"
        if not os.path.exists(bak):
            cv2.imwrite(bak, img)
    for det in detections:
        x1,y1,x2,y2,label,score = det
        cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
        txt = f"{label} {score:.2f}"
        cv2.putText(img, txt, (int(x1), max(10,int(y1)-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)
    cv2.imwrite(img_path, img)
    return img_path

def training_instructions_snippet():
    return """
To train a YOLOv8 model for classes ['door','door_open','door_closed'] using Ultralytics:
1. Prepare dataset in YOLO format (images + labels). Create data.yaml:
   train: path/to/train/images
   val: path/to/val/images
   names: ['door','door_open','door_closed']
2. Train:
   pip install ultralytics
   yolo detect train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640
   (adjust model, epochs, imgsz for performance)
3. After training, save best weights (e.g., runs/detect/train/weights/best.pt) and point the script to that file.
"""

def run_yolo_on_images(coach_meta, weights_path=None, device="cpu", conf=0.35):
    """
    coach_meta: list of coach dicts.
    If `weights_path` exists it will be used. Otherwise fallback heuristic is used.
    Returns list of annotated image paths.
    """
    annotated = []
    # gather images per coach
    images = []
    for meta in coach_meta:
        folder = meta["folder"]
        imgs = [os.path.join(folder,f) for f in os.listdir(folder) if f.lower().endswith(".jpg")]
        imgs = sorted(imgs)
        images.extend(imgs)

    if not images:
        return annotated

    if HAS_ULTRALYTICS and weights_path and os.path.exists(weights_path):
        model = YOLO(weights_path)
        # model.predict handles device internally; set conf threshold
        for img_path in tqdm(images, desc="YOLO infer"):
            results = model(img_path, imgsz=640, device=device, conf=conf)[0]
            dets = []
            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy()
                scores = results.boxes.conf.cpu().numpy()
                for box, cls, score in zip(boxes, classes, scores):
                    x1,y1,x2,y2 = box
                    name = model.names[int(cls)] if hasattr(model, "names") else str(int(cls))
                    dets.append((x1,y1,x2,y2, name, float(score)))
            ap = _annotate_and_save(img_path, dets, backup_original=True)
            if ap:
                annotated.append(ap)
    else:
        # fallback heuristic
        print("[detector] Ultralytics not available or weights missing. Using heuristic contour fallback.")
        for img_path in tqdm(images, desc="Heuristic infer"):
            img = cv2.imread(img_path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            edges = cv2.Canny(blur, 60, 160)
            kern = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
            dil = cv2.dilate(edges, kern, iterations=2)
            cnts, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            dets = []
            h_img,w_img = img.shape[:2]
            for c in cnts:
                x,y,w,h = cv2.boundingRect(c)
                if h > 0.25*h_img and 0.08*w_img < w < 0.3*w_img:
                    roi = gray[y:y+h, x:x+w]
                    mean_int = int(roi.mean())
                    score = 0.6
                    label = "door_closed" if mean_int > 100 else "door_open"
                    dets.append((x,y,x+w,y+h,label,score))
            ap = _annotate_and_save(img_path, dets, backup_original=True)
            if ap:
                annotated.append(ap)

    if not (HAS_ULTRALYTICS and weights_path and os.path.exists(weights_path)):
        print(training_instructions_snippet())
    return annotated