// YOLO Object Tracking - Web Version
// Author: Alejandro Rebolledo

// Configure ONNX Runtime WASM paths to load from CDN
// This ensures it works correctly on GitHub Pages without needing to upload .wasm files
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.16.3/dist/";

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

// Debug logging
function log(msg) {
    const consoleEl = document.getElementById('debug-console');
    if (consoleEl) {
        const div = document.createElement('div');
        div.textContent = `> ${msg}`;
        consoleEl.appendChild(div);
        consoleEl.scrollTop = consoleEl.scrollHeight;
    }
    console.log(msg);
}

// Initialize
async function init() {
    try {
        log('Initializing...');
        statusEl.textContent = 'Loading YOLO model...';
        log(`Loading model from: ${CONFIG.modelPath}`);

        // Check if ort is loaded
        if (typeof ort === 'undefined') {
            throw new Error('onnxruntime-web script not loaded! Check internet connection.');
        }

        session = await ort.InferenceSession.create(CONFIG.modelPath);
        statusEl.textContent = 'Model loaded! Click "Start Camera" to begin.';
        log('Model loaded successfully!');
        log(`Input names: ${session.inputNames}`);
        log(`Output names: ${session.outputNames}`);
        startBtn.disabled = false;
    } catch (error) {
        console.error('Error loading model:', error);
        statusEl.textContent = 'Error loading model.';
        log(`ERROR: ${error.message}`);
        log('TIP: If running locally, use a local server (python -m http.server). Direct file:// access is blocked by browsers.');
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
        // Assign IDs and draw
        assignObjectIds(finalDetections);
        drawDetections(finalDetections);
        updateFPS();

        if (finalDetections.length > 0) {
            // log(`Detected ${finalDetections.length} objects`);
        }

    } catch (error) {
        console.error('Detection error:', error);
        log(`Inference Error: ${error.message}`);
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
