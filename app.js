// YOLO Object Tracking - Web Version
// Author: Alejandro Rebolledo

// Configuration
const CONFIG = {
    modelPath: 'yolo11n.onnx',
    inputSize: 640,
    confidenceThreshold: 0.25,
    iouThreshold: 0.45,
    maxTrailLength: 30,
    movementThreshold: 2.0
};

// COCO class names
const COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
];

// Global state
let session = null;
let stream = null;
let animationId = null;
let trackHistory = {};
let nextObjectId = 0;
let objectIdMap = new Map();
let fpsCounter = { frames: 0, lastTime: Date.now() };
let showTrails = true;

// DOM elements
const webcam = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const statusEl = document.getElementById('status');
const fpsEl = document.getElementById('fps');
const objectCountEl = document.getElementById('objectCount');
const showTrailsCheckbox = document.getElementById('showTrails');

// Initialize
async function init() {
    try {
        statusEl.textContent = 'Loading YOLO model...';
        session = await ort.InferenceSession.create(CONFIG.modelPath);
        statusEl.textContent = 'Model loaded! Click "Start Camera" to begin.';
        startBtn.disabled = false;
    } catch (error) {
        console.error('Error loading model:', error);
        statusEl.textContent = 'Error loading model. Please check console.';
    }
}

// Start webcam
async function startWebcam() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 1280, height: 720, facingMode: 'user' }
        });
        webcam.srcObject = stream;

        webcam.onloadedmetadata = () => {
            canvas.width = webcam.videoWidth;
            canvas.height = webcam.videoHeight;
            startBtn.disabled = true;
            stopBtn.disabled = false;
            statusEl.textContent = 'Running...';
            detectObjects();
        };
    } catch (error) {
        console.error('Error accessing webcam:', error);
        statusEl.textContent = 'Error accessing webcam. Please allow camera access.';
    }
}

// Stop webcam
function stopWebcam() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    if (animationId) {
        cancelAnimationFrame(animationId);
        animationId = null;
    }
    startBtn.disabled = false;
    stopBtn.disabled = true;
    statusEl.textContent = 'Stopped';
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// Preprocess image for YOLO
function preprocessImage(video) {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = CONFIG.inputSize;
    tempCanvas.height = CONFIG.inputSize;
    const tempCtx = tempCanvas.getContext('2d');

    tempCtx.drawImage(video, 0, 0, CONFIG.inputSize, CONFIG.inputSize);
    const imageData = tempCtx.getImageData(0, 0, CONFIG.inputSize, CONFIG.inputSize);

    const float32Data = new Float32Array(3 * CONFIG.inputSize * CONFIG.inputSize);
    for (let i = 0; i < imageData.data.length; i += 4) {
        const idx = i / 4;
        float32Data[idx] = imageData.data[i] / 255.0; // R
        float32Data[CONFIG.inputSize * CONFIG.inputSize + idx] = imageData.data[i + 1] / 255.0; // G
        float32Data[2 * CONFIG.inputSize * CONFIG.inputSize + idx] = imageData.data[i + 2] / 255.0; // B
    }

    return new ort.Tensor('float32', float32Data, [1, 3, CONFIG.inputSize, CONFIG.inputSize]);
}

// Non-Maximum Suppression
function nms(boxes, scores, iouThreshold) {
    const selected = [];
    const indices = scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a]);

    while (indices.length > 0) {
        const current = indices.shift();
        selected.push(current);

        indices.splice(0, indices.length, ...indices.filter(idx => {
            const iou = calculateIoU(boxes[current], boxes[idx]);
            return iou < iouThreshold;
        }));
    }

    return selected;
}

// Calculate Intersection over Union
function calculateIoU(box1, box2) {
    const x1 = Math.max(box1.x, box2.x);
    const y1 = Math.max(box1.y, box2.y);
    const x2 = Math.min(box1.x + box1.w, box2.x + box2.w);
    const y2 = Math.min(box1.y + box1.h, box2.y + box2.h);

    const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const area1 = box1.w * box1.h;
    const area2 = box2.w * box2.h;
    const union = area1 + area2 - intersection;

    return intersection / union;
}

// Assign IDs to detected objects
function assignObjectIds(detections) {
    const currentIds = new Set();

    detections.forEach(det => {
        let bestMatch = null;
        let bestDist = Infinity;

        for (const [id, history] of Object.entries(trackHistory)) {
            if (history.length === 0) continue;
            const lastPos = history[history.length - 1];
            const dist = Math.sqrt(Math.pow(det.cx - lastPos.x, 2) + Math.pow(det.cy - lastPos.y, 2));

            if (dist < 100 && dist < bestDist) {
                bestDist = dist;
                bestMatch = id;
            }
        }

        if (bestMatch) {
            det.id = parseInt(bestMatch);
        } else {
            det.id = nextObjectId++;
            trackHistory[det.id] = [];
        }

        currentIds.add(det.id);
        trackHistory[det.id].push({ x: det.cx, y: det.cy });

        if (trackHistory[det.id].length > CONFIG.maxTrailLength) {
            trackHistory[det.id].shift();
        }
    });

    // Clean up old tracks
    Object.keys(trackHistory).forEach(id => {
        if (!currentIds.has(parseInt(id))) {
            delete trackHistory[id];
        }
    });
}

// Get movement direction
function getDirection(objectId) {
    const history = trackHistory[objectId];
    if (!history || history.length < 2) return 'still';

    const lookback = Math.min(5, history.length - 1);
    const current = history[history.length - 1];
    const past = history[history.length - 1 - lookback];

    const dx = current.x - past.x;
    const dy = current.y - past.y;

    if (Math.abs(dx) > Math.abs(dy)) {
        if (dx > CONFIG.movementThreshold) return 'moving right';
        if (dx < -CONFIG.movementThreshold) return 'moving left';
    } else {
        if (dy > CONFIG.movementThreshold) return 'moving down';
        if (dy < -CONFIG.movementThreshold) return 'moving up';
    }

    return 'still';
}

// Draw detections
function drawDetections(detections) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const scaleX = canvas.width / CONFIG.inputSize;
    const scaleY = canvas.height / CONFIG.inputSize;

    detections.forEach(det => {
        const x = det.x * scaleX;
        const y = det.y * scaleY;
        const w = det.w * scaleX;
        const h = det.h * scaleY;
        const cx = det.cx * scaleX;
        const cy = det.cy * scaleY;

        // Draw trail
        if (showTrails && trackHistory[det.id] && trackHistory[det.id].length > 1) {
            ctx.strokeStyle = 'rgba(230, 230, 230, 0.8)';
            ctx.lineWidth = 2;
            ctx.beginPath();
            const trail = trackHistory[det.id];
            ctx.moveTo(trail[0].x * scaleX, trail[0].y * scaleY);
            for (let i = 1; i < trail.length; i++) {
                ctx.lineTo(trail[i].x * scaleX, trail[i].y * scaleY);
            }
            ctx.stroke();
        }

        // Draw bounding box
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);

        // Draw label
        const direction = getDirection(det.id);
        const label = `id:${det.id} ${det.class} ${direction}`;

        ctx.font = '16px Arial';
        const textMetrics = ctx.measureText(label);
        const textX = cx - textMetrics.width / 2;
        const textY = cy + 8;

        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(textX - 5, textY - 20, textMetrics.width + 10, 25);

        ctx.fillStyle = '#00ff00';
        ctx.fillText(label, textX, textY);
    });

    objectCountEl.textContent = detections.length;
}

// Update FPS
function updateFPS() {
    fpsCounter.frames++;
    const now = Date.now();
    const elapsed = now - fpsCounter.lastTime;

    if (elapsed >= 1000) {
        const fps = Math.round((fpsCounter.frames * 1000) / elapsed);
        fpsEl.textContent = fps;
        fpsCounter.frames = 0;
        fpsCounter.lastTime = now;
    }
}

// Main detection loop
async function detectObjects() {
    if (!session || !stream) return;

    try {
        const inputTensor = preprocessImage(webcam);
        const results = await session.run({ images: inputTensor });
        const output = results.output0.data;

        // Parse YOLO output
        const detections = [];
        const numDetections = output.length / 84; // 84 = 4 (bbox) + 80 (classes)

        for (let i = 0; i < numDetections; i++) {
            const offset = i * 84;
            const cx = output[offset];
            const cy = output[offset + 1];
            const w = output[offset + 2];
            const h = output[offset + 3];

            const scores = Array.from(output.slice(offset + 4, offset + 84));
            const maxScore = Math.max(...scores);
            const classId = scores.indexOf(maxScore);

            if (maxScore > CONFIG.confidenceThreshold) {
                detections.push({
                    x: cx - w / 2,
                    y: cy - h / 2,
                    w, h, cx, cy,
                    score: maxScore,
                    class: COCO_CLASSES[classId] || `class_${classId}`
                });
            }
        }

        // Apply NMS
        const boxes = detections.map(d => ({ x: d.x, y: d.y, w: d.w, h: d.h }));
        const scores = detections.map(d => d.score);
        const selectedIndices = nms(boxes, scores, CONFIG.iouThreshold);
        const finalDetections = selectedIndices.map(i => detections[i]);

        // Assign IDs and draw
        assignObjectIds(finalDetections);
        drawDetections(finalDetections);
        updateFPS();

    } catch (error) {
        console.error('Detection error:', error);
    }

    animationId = requestAnimationFrame(detectObjects);
}

// Event listeners
startBtn.addEventListener('click', startWebcam);
stopBtn.addEventListener('click', stopWebcam);
showTrailsCheckbox.addEventListener('change', (e) => {
    showTrails = e.target.checked;
});

// Initialize on load
init();
