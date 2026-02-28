# 🚙 Offroad Terrain Segmentation System

A complete pixel-wise terrain classification system for offroad images using PyTorch deep learning.

## 📋 Overview

This system analyzes offroad terrain images and classifies every pixel into terrain categories (road, grass, dirt, sand, rocks, water, vegetation, sky, obstacles, etc.). It provides:

- **Pixel-wise classification**: Every pixel is classified into one of 10 terrain categories
- **Visual segmentation**: Color-coded overlay showing terrain distribution
- **Statistical analysis**: Percentage breakdown of terrain types
- **Data export**: Download results as images and CSV files
- **Interactive visualization**: Hover over pixels to see their classification

## 🗂️ Project Structure

```
offroad-segmentation/
├── offroad_best_model_fast.pth    # Your trained PyTorch model
├── offroad_backend.py              # Model inference wrapper
├── api_server.py                   # Flask REST API server
├── offroad_segmentation.html       # Standalone demo (simulated)
├── offroad_segmentation_api.html   # Production frontend (connects to API)
└── README.md                       # This file
```

## 🎯 Model Architecture

The model is a **U-Net** architecture designed for semantic segmentation:

- **Input**: RGB images (3 channels)
- **Output**: 10-class segmentation mask (pixel-wise classification)
- **Architecture**: Encoder-decoder with skip connections
- **Framework**: PyTorch

### Terrain Classes

| ID | Class Name  | Color    | Description           |
|----|-------------|----------|-----------------------|
| 0  | Background  | Black    | Unknown/Background    |
| 1  | Road/Path   | Gray     | Drivable path         |
| 2  | Grass       | Green    | Grassy areas          |
| 3  | Dirt        | Brown    | Dirt terrain          |
| 4  | Sand        | Sandy    | Sandy surfaces        |
| 5  | Rock        | DimGray  | Rocky terrain         |
| 6  | Water       | Blue     | Water bodies          |
| 7  | Vegetation  | Forest   | Dense vegetation      |
| 8  | Sky         | SkyBlue  | Sky regions           |
| 9  | Obstacle    | Red      | Obstacles/Hazards     |

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Install dependencies
pip install torch torchvision
pip install flask flask-cors
pip install pillow numpy
```

### Option 1: Standalone Demo (No API Required)

This version simulates segmentation for demonstration purposes:

```bash
# Simply open in a web browser
open offroad_segmentation.html
```

### Option 2: Production Setup (Real Model Inference)

1. **Start the API Server**:
```bash
python api_server.py
```

The server will start on `http://localhost:5000`

2. **Open the Frontend**:
```bash
open offroad_segmentation_api.html
```

3. **Upload an image and click "Analyze Terrain"**

## 📡 API Documentation

### Base URL
```
http://localhost:5000/api
```

### Endpoints

#### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "num_classes": 10
}
```

#### 2. Get Classes
```http
GET /api/classes
```

**Response:**
```json
{
  "classes": [
    {
      "id": 0,
      "name": "Background",
      "color": "#000000"
    },
    ...
  ]
}
```

#### 3. Segment Image
```http
POST /api/segment
Content-Type: multipart/form-data

file: <image file>
```

**Response:**
```json
{
  "success": true,
  "mask": "<base64 encoded PNG>",
  "statistics": [
    {
      "class_id": 2,
      "class_name": "Grass",
      "pixel_count": 150000,
      "percentage": 45.5,
      "color": "#00FF00"
    },
    ...
  ],
  "image_size": {
    "width": 640,
    "height": 480
  },
  "class_names": {...}
}
```

#### 4. Stream Segmentation (Get Prediction Array)
```http
POST /api/segment/stream
Content-Type: multipart/form-data

file: <image file>
```

**Response:**
```json
{
  "success": true,
  "prediction": [[0, 1, 2, ...], [1, 1, 3, ...], ...],
  "width": 640,
  "height": 480,
  "class_names": {...}
}
```

#### 5. Export as CSV
```http
POST /api/export/csv
Content-Type: multipart/form-data

file: <image file>
```

**Response:** CSV file download with columns: `x, y, class, className, color`

#### 6. Batch Processing
```http
POST /api/process-batch
Content-Type: application/json

{
  "images": ["<base64 image 1>", "<base64 image 2>", ...]
}
```

## 💻 Python Usage Examples

### Basic Usage

```python
from offroad_backend import OffroadSegmenter

# Initialize
segmenter = OffroadSegmenter('offroad_best_model_fast.pth')

# Segment an image
result = segmenter.segment_image('path/to/image.jpg')
prediction = result['prediction']  # 2D numpy array

# Get statistics
stats = segmenter.get_statistics(prediction)
for stat in stats:
    print(f"{stat['class_name']}: {stat['percentage']:.2f}%")

# Create colored visualization
colored_mask = segmenter.create_colored_mask(prediction)
colored_mask.save('output_mask.png')

# Get pixel-level data
pixel_data = segmenter.get_pixel_data(prediction)
# Returns: [{'x': 0, 'y': 0, 'class': 2, 'className': 'Grass', ...}, ...]
```

### Save All Results

```python
# Process and save everything
results = segmenter.save_results(
    'input_image.jpg',
    output_dir='./results'
)

# Results saved:
# - results/segmentation_mask.png
# - results/pixel_data.csv
# - results/statistics.json
```

### Batch Processing

```python
import os
from glob import glob

# Process all images in a folder
image_paths = glob('images/*.jpg')

for img_path in image_paths:
    name = os.path.splitext(os.path.basename(img_path))[0]
    output_dir = f'results/{name}'
    segmenter.save_results(img_path, output_dir)
    print(f"Processed: {img_path}")
```

## 🎨 Frontend Features

### Interactive Visualization
- **Hover over pixels**: See real-time classification
- **Adjustable overlay**: Control transparency (0-100%)
- **Toggle overlay**: Show/hide segmentation mask
- **Zoom support**: Full-resolution pixel inspection

### Export Options
- **PNG Mask**: Download colored segmentation mask
- **CSV Data**: Export complete pixel-wise data
- **Statistics**: View percentage distribution

### Data Output Format

**CSV Structure:**
```csv
x,y,class,className,color
0,0,2,Grass,#00FF00
1,0,2,Grass,#00FF00
2,0,1,Road/Path,#808080
...
```

## ⚙️ Configuration

### Model Configuration

Edit `offroad_backend.py` to customize:

```python
# Change number of classes
model = UNet(n_channels=3, n_classes=10, bilinear=True)

# Change device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Modify class names and colors
self.class_names = {...}
self.class_colors = {...}
```

### API Server Configuration

Edit `api_server.py`:

```python
# Change host/port
app.run(host='0.0.0.0', port=5000)

# Enable CORS for specific domains
CORS(app, resources={r"/api/*": {"origins": ["http://example.com"]}})
```

## 🔧 Troubleshooting

### Model Loading Issues

**Error**: `RuntimeError: Error(s) in loading state_dict`

**Solution**: Ensure your model architecture matches the saved checkpoint. The model expects:
- Input: 3-channel RGB images
- Output: 10 classes
- U-Net architecture with specific layer dimensions

### CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Resize images before processing:
```python
from PIL import Image
img = Image.open('large_image.jpg')
img = img.resize((640, 480))
img.save('resized.jpg')
```

2. Use CPU instead:
```python
segmenter = OffroadSegmenter('model.pth', device='cpu')
```

### API Connection Issues

**Error**: `Failed to fetch` in browser console

**Solutions**:
1. Ensure API server is running: `python api_server.py`
2. Check correct URL: `http://localhost:5000`
3. Verify CORS is enabled in `api_server.py`

## 📊 Performance

### Speed Benchmarks (typical)

| Image Size | GPU (RTX 3090) | CPU (i7-9700K) |
|------------|----------------|----------------|
| 640×480    | ~0.05s         | ~0.8s          |
| 1280×720   | ~0.15s         | ~2.5s          |
| 1920×1080  | ~0.35s         | ~6.0s          |

### Memory Requirements

- **Model size**: ~50 MB
- **RAM**: 2-4 GB
- **VRAM (GPU)**: 1-2 GB for typical images

## 🔐 Security Considerations

For production deployment:

1. **Add authentication**:
```python
from flask_httpauth import HTTPBasicAuth
auth = HTTPBasicAuth()

@auth.verify_password
def verify_password(username, password):
    # Implement your authentication
    pass
```

2. **Rate limiting**:
```python
from flask_limiter import Limiter
limiter = Limiter(app, key_func=get_remote_address)

@app.route('/api/segment')
@limiter.limit("10 per minute")
def segment_image():
    ...
```

3. **Input validation**:
- Check file types
- Limit file sizes
- Validate image dimensions

## 📝 License

This project uses PyTorch models and follows the respective licensing terms.

## 🤝 Contributing

Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

## 📧 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review API documentation
3. Examine console logs
4. Verify all dependencies are installed

## 🎯 Next Steps

1. **Fine-tune the model**: Train on your specific terrain types
2. **Add more classes**: Extend to recognize additional terrain features
3. **Optimize performance**: Quantize model or use TensorRT
4. **Deploy to cloud**: Use AWS Lambda, Google Cloud Functions, or Azure
5. **Mobile app**: Create iOS/Android apps using TensorFlow Lite

---

**Built with ❤️ using PyTorch, Flask, and React**
