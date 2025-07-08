
import cv2
import json
import os
from utils import draw_labeled_box

def draw_boxes(video_path, detections, player_map, view, out_path):
    cap = cv2.VideoCapture(video_path)
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_id >= len(detections):
            break

        for det in detections[frame_id]:
            bbox = det["bbox"]
            img_path = det["img_path"]

            if view == "broadcast":
                label = f"ID_{img_path.split('/')[-1].split('.')[0]}"
            else:
                matched = player_map.get(img_path)
                label = f"ID_{matched.split('/')[-1].split('.')[0]}" if matched else "Unknown"

            draw_labeled_box(frame, bbox, label)

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()
