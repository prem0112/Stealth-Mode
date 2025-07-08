from ultralytics import YOLO
import cv2
import os
import numpy as np

def detect_and_crop_players(video_path, model_path, output_dir):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    all_detections = []

    os.makedirs(output_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.3, iou=0.3)
        detections = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped = frame[y1:y2, x1:x2]
            player_img = cv2.resize(cropped, (64, 128))
            filename = f"{output_dir}/frame{frame_id}_x{x1}_y{y1}.jpg"
            cv2.imwrite(filename, player_img)
            detections.append({
                "frame": frame_id,
                "bbox": [x1, y1, x2, y2],
                "img_path": filename
            })

        all_detections.append(detections)
        frame_id += 1

    cap.release()
    return all_detections
