from ultralytics import YOLO
import cv2
import torch
import numpy as np

# --- Configuration ---
ENABLE_DEPTH = False
# ---------------------

# --- Depth Estimation Setup ---
if ENABLE_DEPTH:
    model_type = "DPT_Hybrid"  # MiDaS v2.1 - Hybrid (medium accuracy, medium inference speed)

    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
# ------------------------------

model = YOLO("yolo11s.pt")

cap = cv2.VideoCapture(0)

# Diccionario para guardar el historial de posiciones por ID
track_history = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # persist=True mantiene el estado del tracker entre frames
    results = model.track(source=frame, persist=True, verbose=False)
    
    # --- Depth Estimation ---
    depth_map = None
    if ENABLE_DEPTH:
        # Transform input for MiDaS
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = midas_transforms(img).to(device)

        # Predict depth
        with torch.no_grad():
            prediction = midas(input_batch)

            # Resize to original resolution
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
    # ------------------------

    r = results[0]
    annotated = r.plot()
    boxes = r.boxes

    if boxes.id is not None:
        ids = boxes.id.int().cpu().tolist()
        xywh = boxes.xywh.cpu().numpy()  # [x_center, y_center, w, h]

        for (x, y, w, h), obj_id in zip(xywh, ids):
            x, y = float(x), float(y)

            depth_text = ""
            if ENABLE_DEPTH and depth_map is not None:
                # Get depth from the bounding box area
                # Ensure coordinates are within bounds
                x1, y1 = int(x - w / 2), int(y - h / 2)
                x2, y2 = int(x + w / 2), int(y + h / 2)

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                # Extract depth region
                if x2 > x1 and y2 > y1:
                    depth_region = depth_map[y1:y2, x1:x2]
                    raw_depth = np.median(depth_region)
                else:
                    raw_depth = 0

                # Simple depth categorization (adjust thresholds as needed)
                # MiDaS output is inverse depth: higher = closer
                depth_text = f" | Depth: {raw_depth:.1f}"

            # Update track history
            if obj_id not in track_history:
                track_history[obj_id] = []
            
            track_history[obj_id].append((float(x), float(y)))
            
            # Limit history length (e.g., last 30 frames)
            if len(track_history[obj_id]) > 30:
                track_history[obj_id].pop(0)

            # Draw the trail
            points = np.hstack(track_history[obj_id]).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [points], isClosed=False, color=(230, 230, 230), thickness=2)

            # Determine direction based on the last few points
            direction_text = "still"
            if len(track_history[obj_id]) >= 2:
                # Compare current position with the position 5 frames ago (or less if not enough history)
                lookback = min(5, len(track_history[obj_id]) - 1)
                px, py = track_history[obj_id][-lookback-1]
                dx, dy = x - px, y - py

                # Umbral mínimo para considerar movimiento
                thr = 2.0

                if abs(dx) > abs(dy):
                    if dx > thr:
                        direction_text = "moving right"
                    elif dx < -thr:
                        direction_text = "moving left"
                else:
                    if dy > thr:
                        direction_text = "moving down"
                    elif dy < -thr:
                        direction_text = "moving up"

            # Dibujar texto de dirección y profundidad
            label = f"id:{obj_id} {direction_text}{depth_text}"
            
            # Calculate text size to center it
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            cv2.putText(
                annotated,
                label,
                (int(x - text_w / 2), int(y + text_h / 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    cv2.imshow("Tracking with direction", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
