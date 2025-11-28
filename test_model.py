"""
YOLO Model Testing Script

Fast testing of ONNX models (Pose and Object detection) using webcam input.
Replicates the logic of the web app for local testing without a browser.

Usage:
    python test_model.py --model pose    # Test pose estimation
    python test_model.py --model object  # Test object detection
    python test_model.py --path custom_model.onnx  # Use custom model

Author: Alejandro Rebolledo (arebolledo@udd.cl)
License: CC BY-NC 4.0
"""

import cv2
import numpy as np
import onnxruntime as ort
import argparse

# Constants
input_shape = (320, 320)
confidence_threshold = 0.25
iou_threshold = 0.45

# COCO Classes
CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Skeleton connections
SKELETON = [
    [5, 7], [7, 9], [6, 8], [8, 10], # Arms
    [5, 6], [5, 11], [6, 12], [11, 12], # Torso
    [11, 13], [13, 15], [12, 14], [14, 16], # Legs
    [0, 1], [0, 2], [1, 3], [2, 4] # Face
]

def preprocess(frame):
    img = cv2.resize(frame, input_shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = img / 255.0
    img = img.astype(np.float32)
    return img

def intersection(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    return max(0, x2 - x1) * max(0, y2 - y1)

def union(box1, box2):
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return box1_area + box2_area - intersection(box1, box2)

def iou(box1, box2):
    return intersection(box1, box2) / union(box1, box2)

def nms(boxes, iou_thresh):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda x: x['score'], reverse=True)
    result = []
    while boxes:
        current = boxes.pop(0)
        result.append(current)
        boxes = [box for box in boxes if iou(current['box'], box['box']) < iou_thresh]
    return result

def process_output(output, img_width, img_height, model_type):
    output = output[0] # Batch size 1
    # Transpose to [N, channels]
    output = output.transpose()
    
    boxes = []
    
    for row in output:
        prob = 0
        class_id = 0
        label = 'person'
        
        if model_type == 'pose':
            # 56 channels: 4 box, 1 conf, 17*3 keypoints
            prob = row[4]
            if prob < confidence_threshold:
                continue
            
            xc, yc, w, h = row[0], row[1], row[2], row[3]
            
            # Keypoints
            keypoints = []
            for k in range(17):
                kx = row[5 + k*3]
                ky = row[5 + k*3 + 1]
                kconf = row[5 + k*3 + 2]
                keypoints.append({
                    'x': int(kx / input_shape[0] * img_width),
                    'y': int(ky / input_shape[1] * img_height),
                    'conf': kconf
                })
                
        else: # object
            # 84 channels: 4 box, 80 classes
            scores = row[4:]
            class_id = np.argmax(scores)
            prob = scores[class_id]
            if prob < confidence_threshold:
                continue
            
            label = CLASSES[class_id]
            xc, yc, w, h = row[0], row[1], row[2], row[3]
            keypoints = None

        # Box coordinates
        x1 = int((xc - w/2) / input_shape[0] * img_width)
        y1 = int((yc - h/2) / input_shape[1] * img_height)
        x2 = int((xc + w/2) / input_shape[0] * img_width)
        y2 = int((yc + h/2) / input_shape[1] * img_height)
        
        boxes.append({
            'box': [x1, y1, x2, y2],
            'score': prob,
            'class_id': class_id,
            'label': label,
            'keypoints': keypoints
        })
        
    return nms(boxes, iou_threshold)

def main():
    parser = argparse.ArgumentParser(description='Test YOLO ONNX models')
    parser.add_argument('--model', type=str, default='pose', choices=['pose', 'object'], help='Model type: pose or object')
    parser.add_argument('--path', type=str, help='Path to ONNX model file')
    args = parser.parse_args()

    model_path = args.path
    if not model_path:
        if args.model == 'pose':
            model_path = 'yolo11n-pose-320.onnx'
        else:
            model_path = 'yolo11n-320.onnx'

    print(f"Loading model: {model_path}")
    
    try:
        session = ort.InferenceSession(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    cap = cv2.VideoCapture(0)
    print("Starting webcam... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        input_tensor = preprocess(frame)
        
        outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
        detections = process_output(outputs[0], width, height, args.model)

        for det in detections:
            box = det['box']
            color = (0, 255, 0)
            
            # Draw Box
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            
            # Draw Label
            label = f"{det['label']} {det['score']:.2f}"
            cv2.putText(frame, label, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw Skeleton
            if det['keypoints']:
                for i, kp in enumerate(det['keypoints']):
                    if kp['conf'] > 0.5:
                        cv2.circle(frame, (kp['x'], kp['y']), 3, (0, 0, 255), -1)
                
                for i, j in SKELETON:
                    kp1 = det['keypoints'][i]
                    kp2 = det['keypoints'][j]
                    if kp1['conf'] > 0.5 and kp2['conf'] > 0.5:
                        cv2.line(frame, (kp1['x'], kp1['y']), (kp2['x'], kp2['y']), (255, 0, 0), 2)

        cv2.imshow('YOLO Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
