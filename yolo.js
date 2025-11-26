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
let detectionFrame = 0;  // Para limitar logging

// DOM elements
const webcam = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const fpsEl = document.getElementById('fps');
const objectsEl = document.getElementById('objects');
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
        log(`Loading model from: ${CONFIG.modelPath}`);

        // Check if ort is loaded
        if (typeof ort === 'undefined') {
            throw new Error('onnxruntime-web script not loaded! Check internet connection.');
        }

        log('Creating ONNX Runtime session...');
        session = await ort.InferenceSession.create(CONFIG.modelPath);

        log('Model loaded successfully!');
        log(`Input names: ${session.inputNames.join(', ')}`);
        log(`Output names: ${session.outputNames.join(', ')}`);

        // Hide loading spinner
        const loadingEl = document.getElementById('loading');
        if (loadingEl) {
            loadingEl.style.display = 'none';
        }

        startBtn.disabled = false;
        log('Ready to start! Click "Start Camera"');
    } catch (error) {
        console.error('Error loading model:', error);
        log(`ERROR: ${error.message}`);
        log('TIP: If running locally, use a local server (python -m http.server). Direct file:// access is blocked by browsers.');
    }
}

// Start webcam
async function startWebcam() {
    try {
        log('Requesting camera access...');
        stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 1280, height: 720, facingMode: 'user' }
        });
        webcam.srcObject = stream;

        webcam.onloadedmetadata = () => {
            canvas.width = webcam.videoWidth;
            canvas.height = webcam.videoHeight;
            startBtn.disabled = true;
            stopBtn.disabled = false;
            log(`Camera ready: ${webcam.videoWidth}x${webcam.videoHeight}`);
            log('Starting detection loop...');
            detectObjects();
        };
    } catch (error) {
        console.error('Error accessing webcam:', error);
        log(`ERROR: ${error.message}`);
        log('Please allow camera access and try again.');
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
    log('Detection stopped');
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

// Parse YOLO output (for YOLOv8/v11 format)
function parseYOLOOutput(outputData, dims, videoWidth, videoHeight) {
    // YOLO output format can be:
    // [1, 85, 8400] - standard format (batch, features, detections)
    // Where features = 4 (x,y,w,h) + 1 (objectness) + 80 (classes)

    const detections = [];
    const numClasses = COCO_CLASSES.length;

    let numDetections, outputLength;

    // Handle different output formats
    if (dims.length === 3) {
        // Format: [batch, features, num_detections]
        numDetections = dims[2];
        outputLength = dims[1];
    } else if (dims.length === 2) {
        // Format: [num_detections, features]
        numDetections = dims[0];
        outputLength = dims[1];
    } else {
        log(`ERROR: Unexpected output shape: ${dims}`);
        return [];
    }

    const scaleX = videoWidth / CONFIG.inputSize;
    const scaleY = videoHeight / CONFIG.inputSize;

    // Iterate through detections
    for (let i = 0; i < Math.min(numDetections, outputData.length / outputLength); i++) {
        let idx;

        if (dims.length === 3) {
            // For [batch, features, detections] format
            idx = i * outputLength;
        } else {
            // For [detections, features] format
            idx = i * outputLength;
        }

        // Extract bounding box (center coordinates)
        const cx = (outputData[idx] || 0) * scaleX;
        const cy = (outputData[idx + 1] || 0) * scaleY;
        const w = (outputData[idx + 2] || 0) * scaleX;
        const h = (outputData[idx + 3] || 0) * scaleY;
        const objectness = outputData[idx + 4] || 0;

        // Skip invalid bounding boxes
        if (w <= 0 || h <= 0) continue;
        if (objectness < CONFIG.confidenceThreshold) continue;

        // Find class with highest confidence
        let maxClassScore = 0;
        let classId = 0;

        for (let j = 0; j < Math.min(numClasses, outputLength - 5); j++) {
            const classScore = outputData[idx + 5 + j] || 0;
            if (classScore > maxClassScore) {
                maxClassScore = classScore;
                classId = j;
            }
        }

        // Final confidence is objectness * class_confidence
        const finalConfidence = objectness * maxClassScore;

        if (finalConfidence >= CONFIG.confidenceThreshold && classId < COCO_CLASSES.length) {
            detections.push({
                cx: cx,
                cy: cy,
                x: cx - w / 2,
                y: cy - h / 2,
                w: w,
                h: h,
                confidence: finalConfidence,
                classId: classId,
                className: COCO_CLASSES[classId] || 'unknown'
            });
        }
    }

    // Apply NMS
    if (detections.length > 1) {
        const boxes = detections.map(d => ({ x: d.x, y: d.y, w: d.w, h: d.h }));
        const scores = detections.map(d => d.confidence);
        const kept = nms(boxes, scores, CONFIG.iouThreshold);
        return kept.map(i => detections[i]);
    }

    return detections;
}

// Draw detections
function drawDetections(detections) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw trails first (background)
    if (showTrails) {
        ctx.strokeStyle = 'rgba(200, 200, 200, 0.3)';
        ctx.lineWidth = 2;

        for (const [id, history] of Object.entries(trackHistory)) {
            if (history.length < 2) continue;

            ctx.beginPath();
            ctx.moveTo(history[0].x, history[0].y);

            for (let i = 1; i < history.length; i++) {
                ctx.lineTo(history[i].x, history[i].y);
            }
            ctx.stroke();
        }
    }

    // Draw bounding boxes and labels
    detections.forEach((det, idx) => {
        // Draw bounding box
        ctx.strokeStyle = '#00FF00';
        ctx.lineWidth = 2;
        ctx.strokeRect(det.x, det.y, det.w, det.h);

        // Draw label background
        const label = `${det.className} ${(det.confidence * 100).toFixed(1)}%`;
        ctx.font = 'bold 14px Arial';
        const textMetrics = ctx.measureText(label);
        const labelHeight = 20;
        const labelWidth = textMetrics.width + 10;

        ctx.fillStyle = '#00FF00';
        ctx.fillRect(det.x, det.y - labelHeight - 5, labelWidth, labelHeight);

        // Draw label text
        ctx.fillStyle = '#000000';
        ctx.font = 'bold 14px Arial';
        ctx.fillText(label, det.x + 5, det.y - 5);

        // Draw object ID and direction
        if (det.id !== undefined) {
            const direction = getDirection(det.id);
            ctx.fillStyle = '#FFFF00';
            ctx.font = 'bold 12px Arial';
            ctx.fillText(`ID: ${det.id}`, det.x, det.y + det.h + 15);
            ctx.fillText(`${direction}`, det.x, det.y + det.h + 30);
        }
    });
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
        // Preprocess the current video frame
        const inputTensor = preprocessImage(webcam);

        // Get input name from session
        const inputName = session.inputNames[0];
        const feeds = {};
        feeds[inputName] = inputTensor;

        // Run inference
        if (detectionFrame === 0) log(`Running inference with input: ${inputName}...`);
        const outputs = await session.run(feeds);

        if (detectionFrame === 0) log(`Output names: ${Object.keys(outputs).join(', ')}`);

        // Get output tensor (first output)
        const outputName = session.outputNames[0];
        const outputData = outputs[outputName];

        if (detectionFrame === 0) log(`Output shape: [${outputData.dims.join(', ')}], data length: ${outputData.data.length}`);

        // Parse detections from YOLO output
        const detections = parseYOLOOutput(outputData.data, outputData.dims, webcam.videoWidth, webcam.videoHeight);

        if (detectionFrame === 0) log(`Parsed ${detections.length} detections`);

        // Assign IDs and draw
        assignObjectIds(detections);
        drawDetections(detections);
        updateFPS();

        if (objectsEl) {
            objectsEl.textContent = `Objetos: ${detections.length}`;
        }

        detectionFrame++;

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
