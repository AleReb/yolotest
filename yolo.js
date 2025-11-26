// YOLO Object Tracking - Web Version
// Adapted from GitHub example for YOLOv8/v11 ONNX

// Configuration
const CONFIG = {
    modelPath: 'yolo11n-320.onnx', // Default to Lite model
    inputSize: 320,                // Default to 320
    confidenceThreshold: 0.25,
    iouThreshold: 0.45,
    maxTrailLength: 30,
    movementThreshold: 2.0
};

// Neon Color Palette
const COLORS = [
    '#00FF00', // Lime
    '#00FFFF', // Cyan
    '#FF00FF', // Magenta
    '#FFFF00', // Yellow
    '#FFA500', // Orange
    '#FF69B4', // Hot Pink
    '#00BFFF', // Deep Sky Blue
    '#ADFF2F', // Green Yellow
    '#FF4500', // Orange Red
    '#7FFF00'  // Chartreuse
];

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
let fpsCounter = { frames: 0, lastTime: Date.now() };
let showTrails = true;
let performanceMode = true; // Default to true
let frameCount = 0;

// DOM elements
const webcam = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const fpsEl = document.getElementById('fps');
const objectsEl = document.getElementById('objects');
const showTrailsCheckbox = document.getElementById('showTrails');
const performanceCheckbox = document.getElementById('performanceMode');

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
async function init(modelName = 'yolo11n-320.onnx', size = 320) {
    try {
        // Stop any existing loop
        if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }

        // Show loading
        const loadingEl = document.getElementById('loading');
        if (loadingEl) loadingEl.style.display = 'flex';

        log(`Switching to model: ${modelName} (${size}x${size})...`);

        // Update config
        CONFIG.modelPath = modelName;
        CONFIG.inputSize = size;

        if (typeof ort === 'undefined') {
            throw new Error('onnxruntime-web script not loaded! Check internet connection.');
        }

        // Configure WASM paths
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.16.3/dist/";

        // Release old session if exists
        if (session) {
            try {
                // session.release(); // Not always available/needed in web
                session = null;
            } catch (e) { console.warn(e); }
        }

        session = await ort.InferenceSession.create(CONFIG.modelPath);

        log('Model loaded successfully!');

        if (loadingEl) loadingEl.style.display = 'none';
        startBtn.disabled = false;

        // If we were already running, restart
        if (stream) {
            detectObjects();
        } else {
            log('Ready to start! Click "Start Camera"');
        }

    } catch (error) {
        console.error('Error loading model:', error);
        log(`ERROR: ${error.message}`);
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

/**
 * Prepare input tensor for YOLOv8/v11
 */
async function prepare_input(video) {
    const canvas = document.createElement("canvas");
    canvas.width = CONFIG.inputSize;
    canvas.height = CONFIG.inputSize;
    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, CONFIG.inputSize, CONFIG.inputSize);
    const imgData = context.getImageData(0, 0, CONFIG.inputSize, CONFIG.inputSize);
    const pixels = imgData.data;

    const red = [], green = [], blue = [];
    for (let index = 0; index < pixels.length; index += 4) {
        red.push(pixels[index] / 255.0);
        green.push(pixels[index + 1] / 255.0);
        blue.push(pixels[index + 2] / 255.0);
    }
    const input = [...red, ...green, ...blue];
    return new ort.Tensor(Float32Array.from(input), [1, 3, CONFIG.inputSize, CONFIG.inputSize]);
}

/**
 * Process YOLO output
 */
function process_output(output, img_width, img_height) {
    let boxes = [];

    // Calculate number of anchors dynamically
    // Output shape is [1, 84, N] -> flattened is 84 * N
    // So N = length / 84
    const numAnchors = output.length / 84;

    for (let index = 0; index < numAnchors; index++) {
        // Find class with max probability
        // The output is transposed: [84, N]
        // [cx, cy, w, h, class0 ... class79]
        // Stride is numAnchors

        const [class_id, prob] = [...Array(80).keys()]
            .map(col => [col, output[numAnchors * (col + 4) + index]])
            .reduce((accum, item) => item[1] > accum[1] ? item : accum, [0, 0]);

        if (prob < CONFIG.confidenceThreshold) {
            continue;
        }

        const label = COCO_CLASSES[class_id];
        const xc = output[index];
        const yc = output[numAnchors + index];
        const w = output[2 * numAnchors + index];
        const h = output[3 * numAnchors + index];

        // Convert to image coordinates
        const x1 = (xc - w / 2) / CONFIG.inputSize * img_width;
        const y1 = (yc - h / 2) / CONFIG.inputSize * img_height;
        const x2 = (xc + w / 2) / CONFIG.inputSize * img_width;
        const y2 = (yc + h / 2) / CONFIG.inputSize * img_height;

        // Store as center coordinates for tracking
        boxes.push({
            x: x1,
            y: y1,
            w: x2 - x1,
            h: y2 - y1,
            cx: (x1 + x2) / 2,
            cy: (y1 + y2) / 2,
            class: label,
            score: prob,
            classId: class_id
        });
    }

    // Sort by probability
    boxes = boxes.sort((box1, box2) => box2.score - box1.score);

    // NMS
    const result = [];
    while (boxes.length > 0) {
        result.push(boxes[0]);
        boxes = boxes.filter(box => iou(boxes[0], box) < 0.7);
    }
    return result;
}

/**
 * IoU Calculation
 */
function iou(box1, box2) {
    return intersection(box1, box2) / union(box1, box2);
}

function union(box1, box2) {
    const box1_area = box1.w * box1.h;
    const box2_area = box2.w * box2.h;
    return box1_area + box2_area - intersection(box1, box2);
}

function intersection(box1, box2) {
    const x1 = Math.max(box1.x, box2.x);
    const y1 = Math.max(box1.y, box2.y);
    const x2 = Math.min(box1.x + box1.w, box2.x + box2.w);
    const y2 = Math.min(box1.y + box1.h, box2.y + box2.h);
    return Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
}

// Assign IDs to detected objects (Tracking)
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

    // Draw trails
    if (showTrails) {
        ctx.lineWidth = 2;

        for (const [id, history] of Object.entries(trackHistory)) {
            if (history.length < 2) continue;

            // Use same color for trail as object if possible, or default grey
            // We don't easily have classId here without storing it in history
            ctx.strokeStyle = 'rgba(230, 230, 230, 0.5)';

            ctx.beginPath();
            ctx.moveTo(history[0].x, history[0].y);
            for (let i = 1; i < history.length; i++) {
                ctx.lineTo(history[i].x, history[i].y);
            }
            ctx.stroke();
        }
    }

    detections.forEach(det => {
        // Get color based on class ID
        const color = COLORS[det.classId % COLORS.length];

        // Draw bounding box
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(det.x, det.y, det.w, det.h);

        // Draw label
        const direction = getDirection(det.id);
        const label = `ID:${det.id} ${det.class} ${Math.round(det.score * 100)}%`;
        const subLabel = direction;

        ctx.font = '16px Arial';
        const textMetrics = ctx.measureText(label);
        const textX = det.x;
        const textY = det.y - 5;

        // Background for text
        ctx.fillStyle = color; // Use same color but maybe transparent?
        ctx.globalAlpha = 0.7;
        ctx.fillRect(textX, textY - 20, textMetrics.width + 10, 45);
        ctx.globalAlpha = 1.0;

        // Text color (Black is usually good on neon colors)
        ctx.fillStyle = '#000000';
        ctx.fillText(label, textX + 5, textY);
        ctx.fillText(subLabel, textX + 5, textY + 20);
    });

    if (objectsEl) objectsEl.textContent = `Objetos: ${detections.length}`;
}

// Update FPS
function updateFPS() {
    fpsCounter.frames++;
    const now = Date.now();
    const elapsed = now - fpsCounter.lastTime;

    if (elapsed >= 1000) {
        const fps = Math.round((fpsCounter.frames * 1000) / elapsed);
        fpsEl.textContent = `FPS: ${fps}`;
        fpsCounter.frames = 0;
        fpsCounter.lastTime = now;
    }
}

// Main detection loop
async function detectObjects() {
    if (!session || !stream) return;

    try {
        const input = await prepare_input(webcam);
        const outputs = await session.run({ images: input });
        const output = outputs[session.outputNames[0]].data;

        let detections = process_output(output, webcam.videoWidth, webcam.videoHeight);

        assignObjectIds(detections);
        drawDetections(detections);
        updateFPS();

    } catch (error) {
        console.error('Detection error:', error);
        log(`Error: ${error.message}`);
    }

    animationId = requestAnimationFrame(detectObjects);
}

// Event listeners
startBtn.addEventListener('click', startWebcam);
stopBtn.addEventListener('click', stopWebcam);
showTrailsCheckbox.addEventListener('change', (e) => {
    showTrails = e.target.checked;
});

if (performanceCheckbox) {
    performanceCheckbox.addEventListener('change', (e) => {
        const isPerformance = e.target.checked;
        if (isPerformance) {
            init('yolo11n-320.onnx', 320);
        } else {
            init('yolo11n.onnx', 640);
        }
    });
}

// Initialize on load
init('yolo11n-320.onnx', 320);
