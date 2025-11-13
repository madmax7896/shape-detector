import "./style.css";
import { SelectionManager } from "./ui-utils.js";
import { EvaluationManager } from "./evaluation-manager.js";

export interface Point {
  x: number;
  y: number;
}

export interface DetectedShape {
  type: "circle" | "triangle" | "rectangle" | "pentagon" | "star";
  confidence: number;
  boundingBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  center: Point;
  area: number;
}

export interface DetectionResult {
  shapes: DetectedShape[];
  processingTime: number;
  imageWidth: number;
  imageHeight: number;
}

export class ShapeDetector {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
  }

  /**
   * MAIN ALGORITHM TO IMPLEMENT
   * Method for detecting shapes in an image
   * @param imageData - ImageData from canvas
   * @returns Promise<DetectionResult> - Detection results
   *
   * TODO: Implement shape detection algorithm here
   */
  async detectShapes(imageData: ImageData): Promise<DetectionResult> {
    const startTime = performance.now();
    const { width, height } = imageData;

    // --- 1. PREPROCESSING ---
    const grayData = this.toGrayscale(imageData);
    const binaryData = this.otsuBinarize(grayData, width, height);

    // --- 2. CONTOUR FINDING ---
    const contours = this.findContours(binaryData, width, height);

    const shapes: DetectedShape[] = [];

    // --- 3. FEATURE EXTRACTION & CLASSIFICATION ---
    for (const blob of contours) {
      // --- 3.a. Filter out noise ---
      if (blob.length < 100) {
        continue;
      }

      // --- 3.b. Calculate features ---
      const area = blob.length;
      const center = this.calculateCentroid(blob);
      const boundingBox = this.calculateBoundingBox(blob);
      const orderedPerimeter = this.traceContour(blob, width, height);
      const perimeterLength = orderedPerimeter.length;

      // Compactness (Circularity). 1.0 = perfect circle.
      const compactness = (4 * Math.PI * area) / (perimeterLength * perimeterLength);

      // Filter out thin lines that passed the area test
      if (compactness < 0.2) {
        continue;
      }

      // --- 4.a. CLASSIFY: Circle ---
      if (compactness > 0.95) {
        shapes.push({
          type: "circle",
          confidence: Math.min(compactness, 1.0),
          boundingBox,
          center,
          area,
        });
        continue; // It's a circle, move to the next contour
      }

      // --- 4.b. CLASSIFY: Polygons ---
      // Use RDP to find vertices
      const epsilon = perimeterLength * 0.042;
      const vertices = this.rdp(orderedPerimeter, epsilon);

      let type: DetectedShape["type"] | null = null;
      let confidence = 0.85; // Base confidence

      // --- 4.c. HYBRID CLASSIFICATION LOGIC ---
      const v = vertices.length;

      if (v === 10) {
        type = "star";
      } else if (v === 5) {
        type = "pentagon";
      } else if (v === 3) {
        // It's *probably* a triangle.
        // But a noisy square can also report 3 vertices.
        // We check compactness:
        // A square is compact (~0.78), a triangle is not (~0.6).
        if (compactness > 0.75) {
          type = "square"; // It was a noisy square
          confidence = 0.7;
        } else {
          type = "triangle";
        }
      } else if (v === 4) {
        // It's a 4-sided shape.
        // Use our robust `isSquare` checker.
        if (this.isSquare(vertices)) {
          type = "square";
          confidence = 0.9;
        } else {
          type = "rectangle";
          confidence = 0.9;
        }
      } else {
        // RDP is noisy and gave another number (e.g., 6, 7, 8).
        // Let's use compactness as the main classifier.
        // Compactness values:
        // Square: ~0.785
        // Rectangle: < 0.75 (gets lower as it gets thinner)
        // Pentagon: ~0.86 (but would likely be caught by v=5)
        // Star: ~0.4 (would likely be caught by v=10)
        
        if (compactness > 0.76) {
          type = "square"; // Most likely a noisy square
          confidence = 0.65;
        } else if (compactness > 0.5) {
          type = "rectangle"; // Most likely a noisy rectangle
          confidence = 0.65;
        }
        // If compactness is very low, it's ignored (e.g., noisy star)
      }


      if (type) {
        shapes.push({
          type,
          confidence,
          boundingBox,
          center,
          area,
        });
      }
    }

    const processingTime = performance.now() - startTime;

    return {
      shapes,
      processingTime,
      imageWidth: imageData.width,
      imageHeight: imageData.height,
    };
  }

  // ===================================================================
  // HELPER METHODS
  // ===================================================================

  // --- 1.a. Grayscale Conversion ---
  private toGrayscale(imageData: ImageData): Uint8ClampedArray {
    const { data, width, height } = imageData;
    const grayData = new Uint8ClampedArray(width * height);
    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      const gray = 0.299 * r + 0.587 * g + 0.114 * b;
      grayData[j] = gray;
    }
    return grayData;
  }

  // --- 1.b. Otsu's Binarization ---
  private otsuBinarize(
    grayData: Uint8ClampedArray,
    width: number,
    height: number
  ): number[][] {
    const pixelCount = width * height;
    const histogram = new Array(256).fill(0);
    let totalIntensity = 0;

    for (let i = 0; i < grayData.length; i++) {
      histogram[grayData[i]]++;
      totalIntensity += grayData[i];
    }

    let sum = 0;
    let wB = 0, wF = 0, mB = 0, mF = 0;
    let maxVariance = 0;
    let optimalThreshold = 0;

    for (let t = 0; t < 256; t++) {
      wB += histogram[t];
      if (wB === 0) continue;
      wF = pixelCount - wB;
      if (wF === 0) break;
      sum += t * histogram[t];
      mB = sum / wB;
      mF = (totalIntensity - sum) / wF;
      const variance = wB * wF * (mB - mF) * (mB - mF);
      if (variance > maxVariance) {
        maxVariance = variance;
        optimalThreshold = t;
      }
    }

    const binaryData: number[][] = [];
    for (let y = 0; y < height; y++) {
      binaryData[y] = [];
      for (let x = 0; x < width; x++) {
        binaryData[y][x] =
          grayData[y * width + x] < optimalThreshold ? 1 : 0;
      }
    }
    return binaryData;
  }

  // --- 2. Contour Finding (Connected-Component Labeling) ---
  private findContours(
    binaryData: number[][],
    width: number,
    height: number
  ): Point[][] {
    const contours: Point[][] = [];
    const visited: boolean[][] = Array(height)
      .fill(false)
      .map(() => Array(width).fill(false));

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (binaryData[y][x] === 1 && !visited[y][x]) {
          const contour = this.floodFill(
            binaryData, x, y, width, height, visited
          );
          contours.push(contour);
        }
      }
    }
    return contours;
  }

  // --- 2.a. Flood Fill (BFS) ---
  private floodFill(
    binaryData: number[][],
    startX: number,
    startY: number,
    width: number,
    height: number,
    visited: boolean[][]
  ): Point[] {
    const blob: Point[] = [];
    const queue: Point[] = [{ x: startX, y: startY }];
    visited[startY][startX] = true;

    while (queue.length > 0) {
      const { x, y } = queue.shift()!;
      blob.push({ x, y });

      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const nx = x + dx;
          const ny = y + dy;
          if (
            nx >= 0 && nx < width && ny >= 0 && ny < height &&
            binaryData[ny][nx] === 1 && !visited[ny][nx]
          ) {
            visited[ny][nx] = true;
            queue.push({ x: nx, y: ny });
          }
        }
      }
    }
    return blob;
  }

  // --- 3.a. Feature Calculators ---
  private calculateCentroid(blob: Point[]): Point {
    let sumX = 0, sumY = 0;
    for (const p of blob) {
      sumX += p.x;
      sumY += p.y;
    }
    return { x: sumX / blob.length, y: sumY / blob.length };
  }

  private calculateBoundingBox(blob: Point[]): DetectedShape["boundingBox"] {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const p of blob) {
      minX = Math.min(minX, p.x);
      minY = Math.min(minY, p.y);
      maxX = Math.max(maxX, p.x);
      maxY = Math.max(maxY, p.y);
    }
    return {
      x: minX,
      y: minY,
      width: maxX - minX + 1,
      height: maxY - minY + 1,
    };
  }

  // --- 3.b. Moore-Neighbor Contour Tracing ---
  private traceContour(
    blob: Point[],
    width: number,
    height: number
  ): Point[] {
    const blobSet = new Set<string>(blob.map((p) => `${p.x},${p.y}`));
    const startPoint = blob.reduce((p1, p2) => {
      if (p1.y < p2.y) return p1;
      if (p1.y > p2.y) return p2;
      return p1.x < p2.x ? p1 : p2;
    });

    const perimeter: Point[] = [];
    let currentPoint = startPoint;
    const neighbors = [
      { x: 0, y: -1 }, { x: 1, y: -1 }, { x: 1, y: 0 }, { x: 1, y: 1 },
      { x: 0, y: 1 }, { x: -1, y: 1 }, { x: -1, y: 0 }, { x: -1, y: -1 },
    ];
    let dir = 0; 

    do {
      perimeter.push(currentPoint);
      let nextPoint: Point | null = null;
      let nextDir = (dir + 7) % 8; 
      for (let i = 0; i < 8; i++) {
        const checkDir = (nextDir + i) % 8;
        const neighbor = {
          x: currentPoint.x + neighbors[checkDir].x,
          y: currentPoint.y + neighbors[checkDir].y,
        };
        if (blobSet.has(`${neighbor.x},${neighbor.y}`)) {
          nextPoint = neighbor;
          dir = checkDir;
          break;
        }
      }
      if (!nextPoint) break;
      currentPoint = nextPoint;
    } while (
      currentPoint.x !== startPoint.x ||
      currentPoint.y !== startPoint.y
    );

    return perimeter;
  }

  // --- 4.b. Ramer-Douglas-Peucker Algorithm ---
  private rdp(points: Point[], epsilon: number): Point[] {
    if (points.length < 3) return points;
    let dmax = 0;
    let index = 0;
    const end = points.length - 1;
    for (let i = 1; i < end; i++) {
      const d = this.pointLineDistance(points[i], points[0], points[end]);
      if (d > dmax) {
        index = i;
        dmax = d;
      }
    }
    if (dmax > epsilon) {
      const recResults1 = this.rdp(points.slice(0, index + 1), epsilon);
      const recResults2 = this.rdp(points.slice(index), epsilon);
      return recResults1.slice(0, -1).concat(recResults2);
    } else {
      return [points[0], points[end]];
    }
  }

  // RDP Helper: Perpendicular distance from a point to a line
  private pointLineDistance(p: Point, p1: Point, p2: Point): number {
    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    const num = Math.abs(dy * p.x - dx * p.y + p2.x * p1.y - p2.y * p1.x);
    const den = Math.sqrt(dy * dy + dx * dx);
    return num / den;
  }

  // RDP Helper: Distance between two points
  private dist(p1: Point, p2: Point): number {
    return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
  }

  // --- 4.c.i. Square vs. Rectangle Classifier ---
  private isSquare(vertices: Point[]): boolean {
    if (vertices.length !== 4) return false;
    const d1 = this.dist(vertices[0], vertices[1]);
    const d2 = this.dist(vertices[1], vertices[2]);
    const d3 = this.dist(vertices[2], vertices[3]);
    const d4 = this.dist(vertices[3], vertices[0]);
    const sides = [d1, d2, d3, d4];
    const minSide = Math.min(...sides);
    const maxSide = Math.max(...sides);
    // Check if all sides are roughly equal (within 20% tolerance)
    if (maxSide / minSide < 1.2) {
      return true; // It's a square
    }
    return false; // It's a rectangle
  }

  loadImage(file: File): Promise<ImageData> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        this.canvas.width = img.width;
        this.canvas.height = img.height;
        this.ctx.drawImage(img, 0, 0);
        const imageData = this.ctx.getImageData(0, 0, img.width, img.height);
        resolve(imageData);
      };
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  }
}

class ShapeDetectionApp {
  private detector: ShapeDetector;
  private imageInput: HTMLInputElement;
  private resultsDiv: HTMLDivElement;
  private testImagesDiv: HTMLDivElement;
  private evaluateButton: HTMLButtonElement;
  private evaluationResultsDiv: HTMLDivElement;
  private selectionManager: SelectionManager;
  private evaluationManager: EvaluationManager;

  constructor() {
    const canvas = document.getElementById(
      "originalCanvas"
    ) as HTMLCanvasElement;
    this.detector = new ShapeDetector(canvas);

    this.imageInput = document.getElementById("imageInput") as HTMLInputElement;
    this.resultsDiv = document.getElementById("results") as HTMLDivElement;
    this.testImagesDiv = document.getElementById(
      "testImages"
    ) as HTMLDivElement;
    this.evaluateButton = document.getElementById(
      "evaluateButton"
    ) as HTMLButtonElement;
    this.evaluationResultsDiv = document.getElementById(
      "evaluationResults"
    ) as HTMLDivElement;

    this.selectionManager = new SelectionManager();
    this.evaluationManager = new EvaluationManager(
      this.detector,
      this.evaluateButton,
      this.evaluationResultsDiv
    );

    this.setupEventListeners();
    this.loadTestImages().catch(console.error);
  }

  private setupEventListeners(): void {
    this.imageInput.addEventListener("change", async (event) => {
      const file = (event.target as HTMLInputElement).files?.[0];
      if (file) {
        await this.processImage(file);
      }
    });

    this.evaluateButton.addEventListener("click", async () => {
      const selectedImages = this.selectionManager.getSelectedImages();
      await this.evaluationManager.runSelectedEvaluation(selectedImages);
    });
  }

  private async processImage(file: File): Promise<void> {
    try {
      this.resultsDiv.innerHTML = "<p>Processing...</p>";

      const imageData = await this.detector.loadImage(file);
      const results = await this.detector.detectShapes(imageData);

      this.displayResults(results);
    } catch (error) {
      this.resultsDiv.innerHTML = `<p>Error: ${error}</p>`;
    }
  }

  private displayResults(results: DetectionResult): void {
    const { shapes, processingTime } = results;

    let html = `
      <p><strong>Processing Time:</strong> ${processingTime.toFixed(2)}ms</p>
      <p><strong>Shapes Found:</strong> ${shapes.length}</p>
    `;

    if (shapes.length > 0) {
      html += "<h4>Detected Shapes:</h4><ul>";
      shapes.forEach((shape) => {
        html += `
          <li>
            <strong>${
              shape.type.charAt(0).toUpperCase() + shape.type.slice(1)
            }</strong><br>
            Confidence: ${(shape.confidence * 100).toFixed(1)}%<br>
            Center: (${shape.center.x.toFixed(1)}, ${shape.center.y.toFixed(
          1
        )})<br>
            Area: ${shape.area.toFixed(1)}px²
          </li>
        `;
      });
      html += "</ul>";
    } else {
      html +=
        "<p>No shapes detected. Please implement the detection algorithm.</p>";
    }

    this.resultsDiv.innerHTML = html;
  }

  private async loadTestImages(): Promise<void> {
    try {
      const module = await import("./test-images-data.js");
      const testImages = module.testImages;
      const imageNames = module.getAllTestImageNames();

      let html =
        '<h4>Click to upload your own image or use test images for detection. Right-click test images to select/deselect for evaluation:</h4><div class="evaluation-controls"><button id="selectAllBtn">Select All</button><button id="deselectAllBtn">Deselect All</button><span class="selection-info">0 images selected</span></div><div class="test-images-grid">';

      // Add upload functionality as first grid item
      html += `
        <div class="test-image-item upload-item" onclick="triggerFileUpload()">
          <div class="upload-icon">📁</div>
          <div class="upload-text">Upload Image</div>
          <div class="upload-subtext">Click to select file</div>
        </div>
      `;

      imageNames.forEach((imageName) => {
        const dataUrl = testImages[imageName as keyof typeof testImages];
        const displayName = imageName
          .replace(/[_-]/g, " ")
          .replace(/\.(svg|png)$/i, "");
        html += `
          <div class="test-image-item" data-image="${imageName}" 
               onclick="loadTestImage('${imageName}', '${dataUrl}')" 
               oncontextmenu="toggleImageSelection(event, '${imageName}')">
            <img src="${dataUrl}" alt="${imageName}">
            <div>${displayName}</div>
          </div>
        `;
      });

      html += "</div>";
      this.testImagesDiv.innerHTML = html;

      this.selectionManager.setupSelectionControls();

      (window as any).loadTestImage = async (name: string, dataUrl: string) => {
        try {
          const response = await fetch(dataUrl);
          const blob = await response.blob();
          const file = new File([blob], name, { type: "image/svg+xml" });

          const imageData = await this.detector.loadImage(file);
          const results = await this.detector.detectShapes(imageData);
          this.displayResults(results);

          console.log(`Loaded test image: ${name}`);
        } catch (error) {
          console.error("Error loading test image:", error);
        }
      };

      (window as any).toggleImageSelection = (
        event: MouseEvent,
        imageName: string
      ) => {
        event.preventDefault();
        this.selectionManager.toggleImageSelection(imageName);
      };

      // Add upload functionality
      (window as any).triggerFileUpload = () => {
        this.imageInput.click();
      };
    } catch (error) {
      this.testImagesDiv.innerHTML = `
        <p>Test images not available. Run 'node convert-svg-to-png.js' to generate test image data.</p>
        <p>SVG files are available in the test-images/ directory.</p>
      `;
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  new ShapeDetectionApp();
});
