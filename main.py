from detect_players import detect_and_crop_players
from match_players import match_players
import json

model_path = "best.pt"
broadcast_detections = detect_and_crop_players("broadcast.mp4", model_path, "output/broadcast")
tacticam_detections = detect_and_crop_players("tacticam.mp4", model_path, "output/tacticam")

mapping = match_players(broadcast_detections, tacticam_detections)

with open("output/player_id_map.json", "w") as f:
    json.dump(mapping, f, indent=2)

print("Mapping complete. Output saved to output/player_id_map.json")
