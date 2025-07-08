import numpy as np
import cv2
from scipy.spatial.distance import cdist

def extract_histogram(img_path):
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()

def match_players(broadcast_detections, tacticam_detections, top_k=1):
    match_dict = {}
    for frame_id, (b_det, t_det) in enumerate(zip(broadcast_detections, tacticam_detections)):
        if not b_det or not t_det:
            continue

        b_features = [extract_histogram(p["img_path"]) for p in b_det]
        t_features = [extract_histogram(p["img_path"]) for p in t_det]

        dist_matrix = cdist(np.array(t_features), np.array(b_features), metric='euclidean')
        for i, row in enumerate(dist_matrix):
            best_idx = row.argmin()
            match_dict[t_det[i]["img_path"]] = b_det[best_idx]["img_path"]

    return match_dict
