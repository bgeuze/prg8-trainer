import {
    HandLandmarker,
    FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

let handLandmarker = undefined;
let webcamRunning = false;
let video = null;
let canvasElement = null;
let canvasCtx = null;
let trainingsData = [];
let currentLabel = null;
let knnClassifier = ml5.KNNClassifier();

// Initialize the HandLandmarker
async function createHandLandmarker() {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 2
    });
}

// Initialize webcam
async function initializeWebcam() {
    video = document.getElementById("webcam");
    canvasElement = document.getElementById("output_canvas");
    canvasCtx = canvasElement.getContext("2d");

    const constraints = {
        video: {
            width: 640,
            height: 480
        }
    };

    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    video.addEventListener("loadedmetadata", () => {
        video.play();
        predictWebcam();
    });
}

// Predict webcam
async function predictWebcam() {
    canvasElement.style.width = video.videoWidth;
    canvasElement.style.height = video.videoHeight;
    canvasElement.width = video.videoWidth;
    canvasElement.height = video.videoHeight;

    let startTimeMs = performance.now();
    const results = handLandmarker.detectForVideo(video, startTimeMs);

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    if (results.landmarks) {
        for (const landmarks of results.landmarks) {
            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
                color: "#FFFFFF",
                lineWidth: 5
            });
            drawLandmarks(canvasCtx, landmarks, {
                color: "#000000",
                lineWidth: 2
            });

            if (currentLabel) {
                const pose = landmarks.flatMap(({ x, y, z }) => [x, y, z ?? 0]);
                trainingsData.push({ pose, label: currentLabel });
                knnClassifier.addExample(pose, currentLabel);
                updateStats();
                currentLabel = null;
            } else if (knnClassifier.getNumLabels() > 0) {
                const pose = landmarks.flatMap(({ x, y, z }) => [x, y, z ?? 0]);
                knnClassifier.classify(pose, (error, result) => {
                    if (!error) {
                        const confidence = result.confidencesByLabel[result.label] * 100;
                        document.getElementById('currentPoseDisplay').innerHTML = 
                            `Herkend gebaar: ${result.label} (${confidence.toFixed(1)}%)`;
                    }
                });
            }
        }
    } else {
        document.getElementById('currentPoseDisplay').textContent = "Geen hand gedetecteerd";
    }

    canvasCtx.restore();

    if (webcamRunning) {
        window.requestAnimationFrame(predictWebcam);
    }
}

// Split data into training and test sets
function splitTrainTest(data, trainRatio = 0.8) {
    // Group data by label
    const groupedData = data.reduce((acc, item) => {
        acc[item.label] = acc[item.label] || [];
        acc[item.label].push(item);
        return acc;
    }, {});

    let trainData = [];
    let testData = [];

    // Split each label's data separately to ensure representation
    Object.entries(groupedData).forEach(([label, items]) => {
        const trainSize = Math.floor(items.length * trainRatio);
        const shuffled = [...items].sort(() => Math.random() - 0.5);
        trainData.push(...shuffled.slice(0, trainSize));
        testData.push(...shuffled.slice(trainSize));
    });

    // Shuffle final sets
    trainData = trainData.sort(() => Math.random() - 0.5);
    testData = testData.sort(() => Math.random() - 0.5);

    return { trainData, testData };
}

// Calculate accuracy with test data
async function calculateAccuracyWithTestData(testData) {
    let correct = 0;
    let total = testData.length;
    let promises = [];

    testData.forEach(data => {
        promises.push(new Promise((resolve) => {
            knnClassifier.classify(data.pose, (error, result) => {
                if (!error && result.label === data.label) {
                    correct++;
                }
                resolve();
            });
        }));
    });

    await Promise.all(promises);
    const accuracy = (correct / total) * 100;
    return { accuracy, correct, total };
}

// Draw confusion matrix
function drawConfusionMatrix(matrix, labels) {
    const canvas = document.getElementById('confusionMatrix');
    const ctx = canvas.getContext('2d');
    const size = canvas.width;
    const cellSize = size / (labels.length + 1);

    // Clear canvas
    ctx.clearRect(0, 0, size, size);
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, size, size);

    // Draw labels and grid
    ctx.font = '16px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#1a73e8';

    // Draw column headers
    labels.forEach((label, i) => {
        ctx.fillText(label, (i + 1.5) * cellSize, 0.5 * cellSize);
    });

    // Draw row headers
    labels.forEach((label, i) => {
        // Save context
        ctx.save();
        // Translate to the position
        ctx.translate(0.5 * cellSize, (i + 1.5) * cellSize);
        // Draw the text
        ctx.fillText(label, 0, 0);
        // Restore context
        ctx.restore();
    });

    // Draw cells with color intensity based on value
    const maxVal = Math.max(...matrix.flat().map(n => n || 0));

    matrix.forEach((row, i) => {
        row.forEach((val, j) => {
            const x = (j + 1) * cellSize;
            const y = (i + 1) * cellSize;
            
            // Draw cell background
            const intensity = maxVal > 0 ? (val || 0) / maxVal : 0;
            ctx.fillStyle = `rgba(26, 115, 232, ${intensity * 0.8})`;
            ctx.fillRect(x, y, cellSize, cellSize);
            
            // Draw cell border
            ctx.strokeStyle = '#e0e0e0';
            ctx.strokeRect(x, y, cellSize, cellSize);
            
            // Draw value
            ctx.fillStyle = intensity > 0.5 ? 'white' : '#1a73e8';
            ctx.fillText(val || '0', x + cellSize / 2, y + cellSize / 2);
        });
    });
}

// Update statistics display
function updateStats() {
    const stats = {};
    trainingsData.forEach(data => {
        stats[data.label] = (stats[data.label] || 0) + 1;
    });

    const statsHtml = Object.entries(stats)
        .map(([label, count]) => `${label}: ${count} samples`)
        .join('<br>');

    document.getElementById('statsDisplay').innerHTML = `
        <h3>Training Statistics</h3>
        <p>${statsHtml}</p>
        <p>Total samples: ${trainingsData.length}</p>
    `;
}

// Save model
function saveModel() {
    const modelData = {
        trainingsData: trainingsData,
        timestamp: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(modelData)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'hand_gesture_model.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Event Listeners
document.addEventListener('DOMContentLoaded', async () => {
    await createHandLandmarker();
    
    document.getElementById('webcamButton').addEventListener('click', () => {
        if (!webcamRunning) {
            webcamRunning = true;
            initializeWebcam();
            document.getElementById('webcamButton').textContent = 'DISABLE WEBCAM';
        } else {
            webcamRunning = false;
            video.srcObject.getTracks().forEach(track => track.stop());
            document.getElementById('webcamButton').textContent = 'ENABLE WEBCAM';
        }
    });

    document.getElementById('trainA').addEventListener('click', () => {
        currentLabel = 'A';
    });

    document.getElementById('trainB').addEventListener('click', () => {
        currentLabel = 'B';
    });

    document.getElementById('trainC').addEventListener('click', () => {
        currentLabel = 'C';
    });

    document.getElementById('trainD').addEventListener('click', () => {
        currentLabel = 'D';
    });

    document.getElementById('evaluateButton').addEventListener('click', async () => {
        if (trainingsData.length === 0) {
            alert('Train eerst wat gebaren voordat je evalueert!');
            return;
        }

        // Use 70/30 split instead of 80/20 to get more test samples
        const { trainData, testData } = splitTrainTest(trainingsData, 0.7);
        
        // Clear and retrain classifier with training data only
        knnClassifier.clearAllLabels();
        trainData.forEach(({ pose, label }) => {
            knnClassifier.addExample(pose, label);
        });

        // Initialize confusion matrix
        const labels = ['A', 'B', 'C', 'D'];
        const matrix = Array(4).fill().map(() => Array(4).fill(0));
        
        // Debug logging
        const distribution = testData.reduce((acc, item) => {
            acc[item.label] = (acc[item.label] || 0) + 1;
            return acc;
        }, {});
        console.log('Test data distribution:', distribution);
        console.log('Total test samples:', testData.length);
        console.log('Total train samples:', trainData.length);

        // Evaluate with test data and build confusion matrix
        let correct = 0;
        const promises = testData.map(data => {
            return new Promise((resolve) => {
                knnClassifier.classify(data.pose, (error, result) => {
                    if (!error && result) {
                        const actualLabelIndex = labels.indexOf(data.label);
                        const predictedLabelIndex = labels.indexOf(result.label);
                        if (actualLabelIndex !== -1 && predictedLabelIndex !== -1) {
                            matrix[actualLabelIndex][predictedLabelIndex]++;
                            if (actualLabelIndex === predictedLabelIndex) {
                                correct++;
                            }
                        }
                        console.log(`Classified ${data.label} as ${result.label}`);
                    } else if (error) {
                        console.error('Classification error:', error);
                    }
                    resolve();
                });
            });
        });

        await Promise.all(promises);

        // Calculate accuracy
        const accuracy = (correct / testData.length) * 100;
        
        // Update display with more detailed information
        document.getElementById('statsDisplay').innerHTML += `
            <h3>Evaluation Results</h3>
            <p>Test Accuracy: ${accuracy.toFixed(1)}%</p>
            <p>Correct: ${correct}/${testData.length}</p>
            <p>Test Set Size: ${testData.length} samples (${Object.entries(distribution).map(([label, count]) => `${label}: ${count}`).join(', ')})</p>
            <p>Training Set Size: ${trainData.length} samples</p>
        `;

        // Draw the confusion matrix
        drawConfusionMatrix(matrix, labels);
    });

    document.getElementById('saveButton').addEventListener('click', saveModel);

    document.getElementById('clearButton').addEventListener('click', () => {
        trainingsData = [];
        knnClassifier.clearAllLabels();
        updateStats();
        document.getElementById('confusionMatrix').getContext('2d').clearRect(
            0, 0, 
            document.getElementById('confusionMatrix').width,
            document.getElementById('confusionMatrix').height
        );
    });
}); 