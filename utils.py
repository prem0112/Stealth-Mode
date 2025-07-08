import cv2
import numpy as np
from scipy.spatial.distance import cdist
import os

def extract_histogram(img_path):
    """Extract normalized HSV histogram from a player image"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error reading {img_path}")
        return np.zeros((256,))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()

def compute_similarity_matrix(features_A, features_B):
    """Compute Euclidean distance matrix between two sets of features"""
    return cdist(np.array(features_A), np.array(features_B), metric='euclidean')

def ensure_dir(path):
    """Create directory if it does not exist"""
    os.makedirs(path, exist_ok=True)

def draw_labeled_box(frame, bbox, label, color=(0, 255, 0)):
    """Draw bounding box with label"""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
