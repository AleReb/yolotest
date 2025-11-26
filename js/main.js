// Add labels
const labels = [
  "person",
  "bicycle",
  "car",
  "motorcycle",
  "airplane",
  "bus",
  "train",
  "truck",
  "boat",
  "traffic light",
  "fire hydrant",
  "stop sign",
  "parking meter",
  "bench",
  "bird",
  "cat",
  "dog",
  "horse",
  "sheep",
  "cow",
  "elephant",
  "bear",
  "zebra",
  "giraffe",
  "backpack",
  "umbrella",
  "handbag",
  "tie",
  "suitcase",
  "frisbee",
  "skis",
  "snowboard",
  "sports ball",
  "kite",
  "baseball bat",
  "baseball glove",
  "skateboard",
  "surfboard",
  "tennis racket",
  "bottle",
  "wine glass",
  "cup",
  "fork",
  "knife",
  "spoon",
  "bowl",
  "banana",
  "apple",
  "sandwich",
  "orange",
  "broccoli",
  "carrot",
  "hot dog",
  "pizza",
  "donut",
  "cake",
  "chair",
  "couch",
  "potted plant",
  "bed",
  "dining table",
  "toilet",
  "tv",
  "laptop",
  "mouse",
  "remote",
  "keyboard",
  "cell phone",
  "microwave",
  "oven",
  "toaster",
  "sink",
  "refrigerator",
  "book",
  "clock",
  "vase",
  "scissors",
  "teddy bear",
  "hair drier",
  "toothbrush",
];

// Declare variables
const numClass = labels.length;
let mySession = null;

// Configs
const modelName = "yolo11n.onnx";
const modelInputShape = [1, 3, 640, 640];
const topk = 100;
const iouThreshold = 0.45;
const scoreThreshold = 0.2;

// wait until opencv.js initialized
cv["onRuntimeInitialized"] = async () => {
  console.log("OpenCV initialized, loading ONNX models...");
  try {
    // create session
    const yolov8 = await ort.InferenceSession.create(`${modelName}`);
    console.log("Models loaded successfully");

    // warmup main model
    const tensor = new ort.Tensor(
      "float32",
      new Float32Array(modelInputShape.reduce((a, b) => a * b)),
      modelInputShape
    );
    await yolov8.run({ images: tensor });
    console.log("Model warmup complete");

    mySession = { net: yolov8 };
    console.log("Session initialized:", mySession);
  } catch (e) {
    console.error("Error initializing models:", e);
  }
};

// Detect Image Function
const detectImage = async (
  image,
  canvas,
  session,
  topk,
  iouThreshold,
  scoreThreshold,
  inputShape
) => {
  try {
    if (!session || !session.net) {
      console.warn("Session not ready");
      return;
    }

    const [modelWidth, modelHeight] = inputShape.slice(2);
    const [input, xRatio, yRatio] = preprocessing(image, modelWidth, modelHeight);

    const tensor = new ort.Tensor("float32", input.data32F, inputShape);
    const outputs = await session.net.run({ images: tensor });

    const boxes = [];

    // YOLOv11 outputs: [1, 84, 8400] -> [batch, features, detections]
    // where features = [x, y, w, h, confidence, class_scores...]
    const output = Object.values(outputs)[0];
    const data = output.data;
    const [, numFeatures, numDetections] = output.dims;

    for (let i = 0; i < numDetections; i++) {
      const offset = i * numFeatures;
      const x = data[offset];
      const y = data[offset + 1];
      const w = data[offset + 2];
      const h = data[offset + 3];
      const confidence = data[offset + 4];

      // Filter by score threshold
      if (confidence < scoreThreshold) continue;

      // Get class scores (everything after confidence)
      let maxScore = 0;
      let maxLabel = 0;
      for (let j = 5; j < numFeatures; j++) {
        const classScore = data[offset + j] * confidence; // multiply by objectness
        if (classScore > maxScore) {
          maxScore = classScore;
          maxLabel = j - 5;
        }
      }

      if (maxScore < scoreThreshold) continue;

      // Convert from model coordinates to image coordinates
      const [x1, y1, width, height] = [
        (x - 0.5 * w) * xRatio,
        (y - 0.5 * h) * yRatio,
        w * xRatio,
        h * yRatio,
      ];

      boxes.push({
        label: maxLabel,
        probability: maxScore,
        bounding: [x1, y1, width, height],
      });
    }

    renderBoxes(canvas, boxes);
    input.delete();
  } catch (e) {
    console.error("Detection error:", e);
  }
};

// Render box
const renderBoxes = (canvas, boxes) => {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); // clean canvas

  const colors = new Colors();

  // font configs
  const font = `${Math.max(
    Math.round(Math.max(ctx.canvas.width, ctx.canvas.height) / 40),
    14
  )}px Arial`;
  ctx.font = font;
  ctx.textBaseline = "top";

  boxes.forEach((box) => {
    const klass = labels[box.label];
    const color = colors.get(box.label);
    const score = (box.probability * 100).toFixed(1);
    const [x1, y1, width, height] = box.bounding;

    // draw box.
    ctx.fillStyle = Colors.hexToRgba(color, 0.2);
    ctx.fillRect(x1, y1, width, height);
    // draw border box
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(
      Math.min(ctx.canvas.width, ctx.canvas.height) / 200,
      2.5
    );
    ctx.strokeRect(x1, y1, width, height);

    // draw the label background.
    ctx.fillStyle = color;
    const textWidth = ctx.measureText(klass + " - " + score + "%").width;
    const textHeight = parseInt(font, 10); // base 10
    const yText = y1 - (textHeight + ctx.lineWidth);
    ctx.fillRect(
      x1 - 1,
      yText < 0 ? 0 : yText,
      textWidth + ctx.lineWidth,
      textHeight + ctx.lineWidth
    );

    // Draw labels
    ctx.fillStyle = "#ffffff";
    ctx.fillText(
      klass + " - " + score + "%",
      x1 - 1,
      yText < 0 ? 1 : yText + 1
    );
  });
};

/**
 * Preprocessing image
 * @param {HTMLImageElement} source image source
 * @param {Number} modelWidth model input width
 * @param {Number} modelHeight model input height
 * @return preprocessed image and configs
 */
const preprocessing = (source, modelWidth, modelHeight) => {
  const mat = cv.imread(source); // read from img tag
  const matC3 = new cv.Mat(mat.rows, mat.cols, cv.CV_8UC3); // new image matrix
  cv.cvtColor(mat, matC3, cv.COLOR_RGBA2BGR); // RGBA to BGR

  // padding image to [n x n] dim
  const maxSize = Math.max(matC3.rows, matC3.cols); // get max size from width and height
  const xPad = maxSize - matC3.cols, // set xPadding
    xRatio = maxSize / matC3.cols; // set xRatio
  const yPad = maxSize - matC3.rows, // set yPadding
    yRatio = maxSize / matC3.rows; // set yRatio
  const matPad = new cv.Mat(); // new mat for padded image
  cv.copyMakeBorder(matC3, matPad, 0, yPad, 0, xPad, cv.BORDER_CONSTANT); // padding black

  const input = cv.blobFromImage(
    matPad,
    1 / 255.0, // normalize
    new cv.Size(modelWidth, modelHeight), // resize to model input size
    new cv.Scalar(0, 0, 0),
    true, // swapRB
    false // crop
  ); // preprocessing image matrix

  // release mat opencv
  mat.delete();
  matC3.delete();
  matPad.delete();

  return [input, xRatio, yRatio];
};

class Colors {
  // ultralytics color palette https://ultralytics.com/
  constructor() {
    this.palette = [
      "#FF3838",
      "#FF9D97",
      "#FF701F",
      "#FFB21D",
      "#CFD231",
      "#48F90A",
      "#92CC17",
      "#3DDB86",
      "#1A9334",
      "#00D4BB",
      "#2C99A8",
      "#00C2FF",
      "#344593",
      "#6473FF",
      "#0018EC",
      "#8438FF",
      "#520085",
      "#CB38FF",
      "#FF95C8",
      "#FF37C7",
    ];
    this.n = this.palette.length;
  }

  get = (i) => this.palette[Math.floor(i) % this.n];
  static hexToRgba = (hex, alpha) => {
    var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result
      ? `rgba(${[
          parseInt(result[1], 16),
          parseInt(result[2], 16),
          parseInt(result[3], 16),
        ].join(", ")}, ${alpha})`
      : null;
  };
}

// Run inference
document.querySelector("#runInference").addEventListener("click", () => {
  if (!mySession) {
    alert("Models are still loading... Please wait a moment and try again.");
    return;
  }

  document.querySelector("#runInference").style.display = "none";

  const video = document.querySelector("#video");
  const canvas = document.querySelector("canvas");
  const context = canvas.getContext("2d");
  video.style.display = "block";

  // Set video stream constraints
  const constraints = {
    audio: false,
    video: { width: 640, height: 480, facingMode: "environment" },
  };

  // Request access to the user's camera
  navigator.mediaDevices
    .getUserMedia(constraints)
    .then((stream) => {
      video.srcObject = stream;
      video.play();
      const detectionInterval = setInterval(() => {
        if (!mySession) {
          clearInterval(detectionInterval);
          return;
        }
        // Draw video frame on canvas
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Run object detection on the canvas image
        detectImage(
          canvas,
          canvas,
          mySession,
          topk,
          iouThreshold,
          scoreThreshold,
          modelInputShape
        );
      }, 100);
    })
    .catch((err) => {
      console.error(err);
    });
  setTimeout(() => {
    document.querySelector("#stopInference").style.display = "block";
  }, 2000);

})

// Stop inference
document.querySelector("#stopInference").addEventListener("click", () => {
  const video = document.querySelector("#video");
  video.style.display = "none";
  let stream = video.srcObject;
  stream.getTracks().forEach(function (track) {    
      track.stop();    
  })
  setTimeout(() => {
    document.querySelector("#stopInference").style.display = "none";
    document.querySelector("#runInference").style.display = "block";
  }, 2000);


})