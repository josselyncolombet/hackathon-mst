const tf = require('@tensorflow/tfjs-node');
const { createCanvas, loadImage } = require('canvas');
const fs = require('fs');
const path = require('path');

// Constants
const MODEL_URL = 'https://storage.googleapis.com/tfjs-models/savedmodel/ssd_mobilenet_v2/model.json';
const FRAMES_DIR = './frames';
const OUTPUT_DIR = './output';
const BALL_CLASS_ID = 37; // Class ID for 'sports ball' in COCO dataset (includes basketball)
const CONFIDENCE_THRESHOLD = 0.30; // Increased threshold to reduce false positives
const SPECIFIC_FRAME = 0; // Set to 0 to process all frames, or a specific frame number to only process that frame

// Ensure output directory exists
if (!fs.existsSync(OUTPUT_DIR)) {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

// Load TensorFlow model
async function loadModel() {
  console.log('Loading model...');
  try {
    const model = await tf.loadGraphModel(MODEL_URL);
    console.log('Model loaded successfully');
    return model;
  } catch (error) {
    console.error('Error loading model:', error);
    throw error;
  }
}

// Process an image and detect basketballs
async function detectBasketball(model, imagePath) {
  try {
    console.log(`Processing: ${path.basename(imagePath)}`);
    
    // Load image
    const image = await loadImage(imagePath);
    
    // Create canvas
    const canvas = createCanvas(image.width, image.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0);
    
    // Convert to tensor
    const input = tf.browser.fromPixels(canvas)
      .expandDims(0);
    
    // Run model prediction
    const result = await model.executeAsync(input);
    
    // Debug output structure
    console.log(`Model returned ${result.length} tensors with shapes:`);
    result.forEach((tensor, i) => {
      console.log(`- Tensor ${i}: shape=${tensor.shape}`);
    });
    
    // Get outputs - try different tensor arrangements
    let boxes, scores, classes;
    
    try {
      // Try first arrangement (most common)
      boxes = result[0].arraySync()[0];
      scores = result[1].arraySync()[0];
      classes = result[2].arraySync()[0];
    } catch (error) {
      console.log("First tensor arrangement failed, trying alternative...");
      try {
        // Try alternative arrangement
        boxes = result[1].arraySync()[0];
        scores = result[2].arraySync()[0];
        classes = result[3].arraySync()[0];
      } catch (error) {
        console.log("Second tensor arrangement failed, trying final alternative...");
        try {
          // One more possible arrangement
          const predictions = result[0].arraySync()[0];
          // Format might be [y1, x1, y2, x2, score, class]
          boxes = predictions.map(pred => pred.slice(0, 4));
          scores = predictions.map(pred => pred[4]);
          classes = predictions.map(pred => pred[5]);
        } catch (error) {
          console.error("All tensor arrangements failed. Debug output:");
          result.forEach((tensor, i) => {
            try {
              console.log(`Tensor ${i} sample:`, tensor.arraySync().slice(0, 1));
            } catch (e) {
              console.log(`Tensor ${i} cannot be converted to array:`, e.message);
            }
          });
          throw new Error("Could not parse model output");
        }
      }
    }
    
    // Find basketball detections
    const detections = [];
    
    // Print first few boxes to debug
    console.log("First few boxes:", boxes.slice(0, 3));
    console.log("First few scores:", scores.slice(0, 3));
    console.log("First few classes:", classes.slice(0, 3));
    
    for (let i = 0; i < scores.length; i++) {
      const classId = typeof classes[i] === 'number' ? Math.round(classes[i]) : classes[i];
      if (scores[i] > CONFIDENCE_THRESHOLD && classId === BALL_CLASS_ID) {
        // Box format might be [ymin, xmin, ymax, xmax] or [y1, x1, y2, x2]
        let box = boxes[i];
        // Make sure box is an array not a single value
        if (!Array.isArray(box)) {
          console.warn(`Box at index ${i} is not an array:`, box);
          continue;
        }
        
        // Extract coordinates - use destructuring if possible, otherwise access by index
        let ymin, xmin, ymax, xmax;
        if (box.length === 4) {
          [ymin, xmin, ymax, xmax] = box;
        } else {
          console.warn(`Unexpected box format at index ${i}:`, box);
          continue;
        }
        
        detections.push({
          box: {
            x: Math.round(xmin * image.width),
            y: Math.round(ymin * image.height),
            width: Math.round((xmax - xmin) * image.width),
            height: Math.round((ymax - ymin) * image.height)
          },
          score: scores[i]
        });
      }
    }
    
    // Clean up tensors
    tf.dispose(result);
    tf.dispose(input);
    
    // Draw detections on the image
    if (detections.length > 0) {
      console.log(`Found ${detections.length} basketball(s) in ${path.basename(imagePath)}`);
      
      // Draw bounding boxes
      for (const detection of detections) {
        const { x, y, width, height } = detection.box;
        console.log(`Basketball: (${(detection.score * 100).toFixed(2)}% confidence)`);
        console.log(`Position: x=${x}, y=${y}, width=${width}, height=${height}`);
        
        // Draw rectangle
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, width, height);
        
        // Draw label
        ctx.fillStyle = 'red';
        ctx.font = '16px Arial';
        ctx.fillText(`Basketball ${(detection.score * 100).toFixed(0)}%`, x, y > 20 ? y - 5 : y + height + 20);
      }
      
      // Save the annotated image
      const outputPath = path.join(OUTPUT_DIR, 'detected_' + path.basename(imagePath));
      const outStream = fs.createWriteStream(outputPath);
      const pngStream = canvas.createPNGStream();
      pngStream.pipe(outStream);
      
      return new Promise((resolve, reject) => {
        outStream.on('finish', () => {
          console.log(`Saved annotated image to: ${outputPath}`);
          resolve(detections);
        });
        outStream.on('error', reject);
      });
    } else {
      console.log(`No basketballs detected in ${path.basename(imagePath)}`);
      return [];
    }
  } catch (error) {
    console.error(`Error processing frame ${imagePath}:`, error);
    return [];
  }
}

// Fallback circular object detection when ML fails
async function detectCircularObjects(imagePath) {
  try {
    console.log("Using fallback circular object detection...");
    
    // Load image
    const image = await loadImage(imagePath);
    
    // Create canvas
    const canvas = createCanvas(image.width, image.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0);
    
    // Get image data for processing
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    // Simple edge detection
    const edges = new Uint8Array(canvas.width * canvas.height);
    const threshold = 30; // Edge detection threshold
    
    // Basic edge detection using brightness changes
    for (let y = 1; y < canvas.height - 1; y++) {
      for (let x = 1; x < canvas.width - 1; x++) {
        const idx = (y * canvas.width + x) * 4;
        const idxUp = ((y - 1) * canvas.width + x) * 4;
        const idxDown = ((y + 1) * canvas.width + x) * 4;
        const idxLeft = (y * canvas.width + (x - 1)) * 4;
        const idxRight = (y * canvas.width + (x + 1)) * 4;
        
        // Calculate brightness of current and surrounding pixels
        const brightness = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        const brightUp = (data[idxUp] + data[idxUp + 1] + data[idxUp + 2]) / 3;
        const brightDown = (data[idxDown] + data[idxDown + 1] + data[idxDown + 2]) / 3;
        const brightLeft = (data[idxLeft] + data[idxLeft + 1] + data[idxLeft + 2]) / 3;
        const brightRight = (data[idxRight] + data[idxRight + 1] + data[idxRight + 2]) / 3;
        
        // Check for edge by comparing brightness differences
        const diff = Math.max(
          Math.abs(brightness - brightUp),
          Math.abs(brightness - brightDown),
          Math.abs(brightness - brightLeft),
          Math.abs(brightness - brightRight)
        );
        
        edges[y * canvas.width + x] = diff > threshold ? 255 : 0;
      }
    }
    
    // Find potential circular objects using a very simple approach
    // Look for regions with high edge density
    const regions = [];
    const visited = new Uint8Array(canvas.width * canvas.height);
    
    // Look for high-edge-density regions
    for (let y = 0; y < canvas.height; y += 10) { // Sample every 10 pixels for speed
      for (let x = 0; x < canvas.width; x += 10) {
        if (visited[y * canvas.width + x]) continue;
        
        // Count edges in a 50x50 region
        let edgeCount = 0;
        let minX = x, maxX = x, minY = y, maxY = y;
        
        for (let dy = -25; dy <= 25; dy++) {
          for (let dx = -25; dx <= 25; dx++) {
            const nx = x + dx;
            const ny = y + dy;
            if (nx >= 0 && nx < canvas.width && ny >= 0 && ny < canvas.height) {
              if (edges[ny * canvas.width + nx] > 0) {
                edgeCount++;
                minX = Math.min(minX, nx);
                maxX = Math.max(maxX, nx);
                minY = Math.min(minY, ny);
                maxY = Math.max(maxY, ny);
              }
              visited[ny * canvas.width + nx] = 1;
            }
          }
        }
        
        // If enough edges found, consider it a potential object
        if (edgeCount > 100) {
          const width = maxX - minX;
          const height = maxY - minY;
          
          // Only keep somewhat circular regions (aspect ratio close to 1)
          const aspectRatio = width / height;
          if (aspectRatio > 0.8 && aspectRatio < 1.2 && 
              width > 30 && height > 30 && width < 200 && height < 200) {
            regions.push({
              box: {
                x: minX,
                y: minY,
                width: width,
                height: height
              },
              score: edgeCount / 500, // Simple score based on edge density
              circularity: Math.min(width, height) / Math.max(width, height) // How circular the region is
            });
          }
        }
      }
    }
    
    // Sort by circularity (most circular first) and keep only the best candidate
    regions.sort((a, b) => b.circularity - a.circularity);
    const filteredRegions = regions.length > 0 ? [regions[0]] : [];
    
    // Visualize the results
    if (filteredRegions.length > 0) {
      console.log(`Found ${filteredRegions.length} potential basketball`);
      
      // Draw regions on canvas
      ctx.strokeStyle = 'blue';
      ctx.lineWidth = 2;
      
      for (const region of filteredRegions) {
        const { x, y, width, height } = region.box;
        ctx.strokeRect(x, y, width, height);
        
        // Draw label
        ctx.fillStyle = 'blue';
        ctx.font = '14px Arial';
        ctx.fillText(`Potential basketball`, x, y > 20 ? y - 5 : y + height + 20);
      }
      
      // Save the annotated image
      const outputPath = path.join(OUTPUT_DIR, 'fallback_' + path.basename(imagePath));
      const outStream = fs.createWriteStream(outputPath);
      const pngStream = canvas.createPNGStream();
      pngStream.pipe(outStream);
      
      return new Promise((resolve, reject) => {
        outStream.on('finish', () => {
          console.log(`Saved fallback detection to: ${outputPath}`);
          resolve(filteredRegions);
        });
        outStream.on('error', reject);
      });
    } else {
      console.log('No circular objects detected by fallback method');
      return [];
    }
  } catch (error) {
    console.error('Error in fallback detection:', error);
    return [];
  }
}

// Extract detections from TensorFlow model output
async function extractDetections(result) {
  try {
    // Check if result is an array of tensors
    if (!Array.isArray(result) || result.length === 0) {
      console.error("Invalid model output format");
      return null;
    }
    
    // Try different output formats based on common SSD MobileNet patterns
    let boxes, scores, classes, numDetections;
    
    try {
      // Standard SSD MobileNet v2 format
      boxes = result[1].arraySync()[0];
      scores = result[2].arraySync()[0];
      classes = result[3].arraySync()[0];
      numDetections = result[0].arraySync()[0];
      console.log("Extracted detections using standard format");
    } catch (error) {
      console.log("Standard format failed, trying alternative format...");
      try {
        // Alternative format
        boxes = result[0].arraySync()[0];
        scores = result[1].arraySync()[0];
        classes = result[2].arraySync()[0];
        console.log("Extracted detections using alternative format");
      } catch (error) {
        console.log("Alternative format failed, trying final format...");
        try {
          // Last resort format
          const detections = result[0].arraySync()[0];
          // Format might be [batch, num_detections, 6] where last dim is [y1, x1, y2, x2, score, class]
          boxes = detections.map(d => [d[0], d[1], d[2], d[3]]);
          scores = detections.map(d => d[4]);
          classes = detections.map(d => d[5]);
          console.log("Extracted detections using final format");
        } catch (error) {
          console.error("All extraction methods failed:", error);
          return null;
        }
      }
    }
    
    return {
      boxes: boxes,
      scores: scores,
      classes: classes.map(c => Math.round(c)), // Ensure classes are integers
    };
  } catch (error) {
    console.error("Error extracting detections:", error);
    return null;
  }
}

// Main function
async function detectBasketballs(frameNumber) {
  try {
    const imagePath = path.join(FRAMES_DIR, `${frameNumber}.png`);
    console.log(`\nProcessing frame ${frameNumber}: ${imagePath}`);
    
    // Check if file exists
    if (!fs.existsSync(imagePath)) {
      console.error(`File not found: ${imagePath}`);
      return;
    }

    // Load the image
    const image = await loadImage(imagePath);
    
    // Create a canvas
    const canvas = createCanvas(image.width, image.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0);
    
    // Convert to tensor for model input
    let tensorInput;
    try {
      tensorInput = tf.browser.fromPixels(canvas).expandDims(0);
    } catch (err) {
      console.error("Error converting image to tensor:", err);
      // Try fallback method if tensor conversion fails
      await detectCircularObjects(imagePath);
      return;
    }
    
    // Run inference
    const startTime = Date.now();
    console.log("Running model inference...");
    let detection;
    try {
      const model = await tf.loadGraphModel(MODEL_URL);
      const result = await model.executeAsync(tensorInput);
      
      // Debug: log the result tensors
      console.log(`Model returned ${result.length} tensors`);
      for (let i = 0; i < result.length; i++) {
        console.log(`Tensor ${i} shape:`, result[i].shape);
      }
      
      // Try different output extraction methods
      detection = await extractDetections(result);
      
      // Clean up tensors
      result.forEach(tensor => tensor.dispose());
    } catch (error) {
      console.error("Error running model:", error);
      tensorInput.dispose();
      // Try fallback method if model inference fails
      await detectCircularObjects(imagePath);
      return;
    }
    
    // Dispose input tensor
    tensorInput.dispose();
    
    console.log(`Inference completed in ${Date.now() - startTime}ms`);
    
    if (!detection || !detection.boxes || !detection.scores || !detection.classes) {
      console.log("Detection failed or returned invalid format");
      // Try fallback method
      await detectCircularObjects(imagePath);
      return;
    }
    
    // Filter detections for basketballs (class 37 in COCO)
    const basketballs = [];
    const numDetections = detection.scores.length;
    
    console.log(`Found ${numDetections} total detections`);
    
    // First, find the highest confidence basketball
    let highestConfidence = 0;
    let bestBallIndex = -1;

    for (let i = 0; i < numDetections; i++) {
      if (detection.classes[i] === BALL_CLASS_ID && detection.scores[i] >= CONFIDENCE_THRESHOLD) {
        if (detection.scores[i] > highestConfidence) {
          highestConfidence = detection.scores[i];
          bestBallIndex = i;
        }
      }
    }

    // Only keep the best basketball detection
    if (bestBallIndex !== -1) {
      basketballs.push({
        box: detection.boxes[bestBallIndex],
        score: detection.scores[bestBallIndex]
      });
    }

    console.log(`Found ${basketballs.length} basketballs`);
    
    if (basketballs.length === 0) {
      console.log("No basketballs detected with ML model, trying fallback method");
      // Try fallback detection if no basketballs found
      await detectCircularObjects(imagePath);
      return;
    }
    
    // Draw bounding boxes
    ctx.lineWidth = 3;
    ctx.strokeStyle = 'red';
    ctx.fillStyle = 'red';
    ctx.font = '16px Arial';
    
    basketballs.forEach(ball => {
      const { box, score } = ball;
      const [y1, x1, y2, x2] = box;
      
      // Coordinates in normalized format (0-1)
      if (y1 >= 0 && y1 <= 1 && x1 >= 0 && x1 <= 1 && y2 >= 0 && y2 <= 1 && x2 >= 0 && x2 <= 1) {
        const xStart = x1 * canvas.width;
        const yStart = y1 * canvas.height;
        const width = (x2 - x1) * canvas.width;
        const height = (y2 - y1) * canvas.height;
        
        ctx.strokeRect(xStart, yStart, width, height);
        ctx.fillText(`Basketball: ${Math.round(score * 100)}%`, xStart, yStart > 20 ? yStart - 5 : yStart + height + 20);
      } else if (Array.isArray(box) && box.length === 4) {
        // Handle non-normalized coordinates or different format
        console.log("Warning: Unexpected box format, attempting to use raw values", box);
        try {
          const [y1, x1, y2, x2] = box;
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
          ctx.fillText(`Basketball: ${Math.round(score * 100)}%`, x1, y1 > 20 ? y1 - 5 : y2 + 20);
        } catch (e) {
          console.error("Failed to draw box:", e);
        }
      } else {
        console.error("Invalid box format:", box);
      }
    });
    
    // Save the output image
    const outputPath = path.join(OUTPUT_DIR, `${frameNumber}.png`);
    const outStream = fs.createWriteStream(outputPath);
    const pngStream = canvas.createPNGStream();
    pngStream.pipe(outStream);
    
    return new Promise((resolve, reject) => {
      outStream.on('finish', () => {
        console.log(`Output saved to: ${outputPath}`);
        resolve();
      });
      outStream.on('error', reject);
    });
  } catch (error) {
    console.error('Error in detectBasketballs:', error);
    // Try fallback method as last resort
    try {
      const imagePath = path.join(FRAMES_DIR, `${frameNumber}.png`);
      await detectCircularObjects(imagePath);
    } catch (fallbackError) {
      console.error('Fallback detection also failed:', fallbackError);
    }
  }
}

// Start processing
async function main() {
  try {
    // Create output directory if it doesn't exist
    if (!fs.existsSync(OUTPUT_DIR)) {
      fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    }
    
    // Get all frame files
    let frameFiles = fs.readdirSync(FRAMES_DIR)
      .filter(file => /\d+\.png$/.test(file))
      .sort((a, b) => {
        const numA = parseInt(a.match(/\d+/)[0]);
        const numB = parseInt(b.match(/\d+/)[0]);
        return numA - numB;
      });
    
    console.log(`Found ${frameFiles.length} frame(s) to process`);
    
    // Process each frame or just the specific one
    if (SPECIFIC_FRAME > 0 && SPECIFIC_FRAME <= frameFiles.length) {
      const specificFrame = SPECIFIC_FRAME;
      const framePath = path.join(FRAMES_DIR, frameFiles[specificFrame - 1]);
      console.log(`Processing only frame ${specificFrame}: ${path.basename(framePath)}`);
      await detectBasketballs(specificFrame);
    } else {
      // Process all frames
      for (const frameFile of frameFiles) {
        const frameNumber = parseInt(frameFile.match(/\d+/)[0]);
        await detectBasketballs(frameNumber);
        console.log('-'.repeat(50));
      }
    }
    
    console.log("Processing complete!");
  } catch (error) {
    console.error("Error in main:", error);
  }
}

// Start processing
main(); 