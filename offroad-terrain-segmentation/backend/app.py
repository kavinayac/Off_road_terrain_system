"""
Offroad Segmentation Flask API Server (FINAL)
"""

import os
import tempfile
import base64
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from offroad_backend import OffroadSegmenter

MODEL_PATH = "offroad_best_model_fast.pth"

app = Flask(__name__)
CORS(app)

print("🚀 Initializing Offroad Segmentation API Server...")

try:
    segmenter = OffroadSegmenter(MODEL_PATH)
    print("✅ API Server ready!")
except Exception as e:
    print("❌ Failed to load model:", e)
    segmenter = None


# ==========================================
# ❤️ HEALTH
# ==========================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": segmenter is not None,
        "device": segmenter.device if segmenter else "none",
        "num_classes": segmenter.num_classes if segmenter else 0
    })


# ==========================================
# 📊 CLASSES
# ==========================================
@app.route("/api/classes", methods=["GET"])
def get_classes():
    classes = []
    for cid, name in segmenter.class_names.items():
        color = "#{:02x}{:02x}{:02x}".format(
            *segmenter.class_colors[cid]
        )
        classes.append({
            "id": cid,
            "name": name,
            "color": color
        })
    return jsonify({"classes": classes})


# ==========================================
# 🧠 SEGMENT
# ==========================================
@app.route("/api/segment", methods=["POST"])
def segment_image():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400

        file = request.files["file"]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            file.save(tmp.name)
            temp_path = tmp.name

        # Run model
        result = segmenter.segment_image(temp_path)
        prediction = result["prediction"]

        # Create mask
        mask_img = segmenter.create_colored_mask(prediction)

        buffered = io.BytesIO()
        mask_img.save(buffered, format="PNG")
        mask_base64 = base64.b64encode(buffered.getvalue()).decode()

        stats = segmenter.get_statistics(prediction)

        return jsonify({
            "success": True,
            "mask": mask_base64,
            "statistics": stats,
            "image_size": {
                "width": prediction.shape[1],
                "height": prediction.shape[0],
            },
        })

    except Exception as e:
        print("❌ Segmentation error:", e)
        return jsonify({"success": False, "error": str(e)}), 500


# ==========================================
# 🚀 RUN
# ==========================================
if __name__ == "__main__":
    print("\n🚀 Server running at http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
