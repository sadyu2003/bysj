import io
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from flask import Flask, jsonify, render_template, request, send_file, url_for
from PIL import Image
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
ANNOTATED_DIR = BASE_DIR / "outputs" / "annotated"
REPORT_DIR = BASE_DIR / "outputs" / "reports"
MODEL_PATH = os.environ.get("PEST_MODEL_PATH", "yolov8m.pt")

for folder in [UPLOAD_DIR, ANNOTATED_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

app = Flask(
    __name__,
    template_folder=str(Path(__file__).resolve().parent / "templates"),
    static_folder=str(Path(__file__).resolve().parent / "static"),
)


def get_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_model() -> YOLO:
    model = YOLO(MODEL_PATH)
    model.to(get_device())
    return model


MODEL = load_model()


@app.route("/")
def index():
    return render_template("index.html", gpu=get_device(), model_path=MODEL_PATH)


@app.route("/api/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "未收到图片文件"}), 400

    file = request.files["image"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{timestamp}_{file.filename}"
    source_path = UPLOAD_DIR / filename
    file.save(source_path)

    results = MODEL.predict(
        source=str(source_path),
        device=get_device(),
        conf=float(request.form.get("conf", 0.25)),
        iou=float(request.form.get("iou", 0.45)),
        imgsz=int(request.form.get("imgsz", 960)),
        verbose=False,
    )

    result = results[0]
    plotted = result.plot()
    annotated_name = f"annotated_{filename}"
    annotated_path = ANNOTATED_DIR / annotated_name
    Image.fromarray(plotted[..., ::-1]).save(annotated_path)

    counts = {}
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls.item())
        cls_name = result.names[cls_id]
        conf = float(box.conf.item())
        xyxy = [round(v, 2) for v in box.xyxy.cpu().numpy().reshape(-1).tolist()]

        counts[cls_name] = counts.get(cls_name, 0) + 1
        detections.append({"class": cls_name, "confidence": round(conf, 4), "bbox": xyxy})

    df = pd.DataFrame(
        [{"class": k, "count": v} for k, v in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]
    )
    report_path = REPORT_DIR / f"report_{timestamp}.csv"
    df.to_csv(report_path, index=False, encoding="utf-8-sig")

    return jsonify(
        {
            "device": get_device(),
            "input_image": url_for("uploaded_file", filename=filename),
            "annotated_image": url_for("annotated_file", filename=annotated_name),
            "counts": counts,
            "detections": detections,
            "report": url_for("report_file", filename=report_path.name),
        }
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_file(UPLOAD_DIR / filename)


@app.route("/outputs/annotated/<path:filename>")
def annotated_file(filename):
    return send_file(ANNOTATED_DIR / filename)


@app.route("/outputs/reports/<path:filename>")
def report_file(filename):
    return send_file(REPORT_DIR / filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
