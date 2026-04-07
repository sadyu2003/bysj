import json
import os
import sqlite3
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
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "pest_system.db"
MODEL_PATH = os.environ.get("PEST_MODEL_PATH", "yolov8m.pt")

for folder in [UPLOAD_DIR, ANNOTATED_DIR, REPORT_DIR, DATA_DIR]:
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


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS detect_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                source_image TEXT NOT NULL,
                annotated_image TEXT NOT NULL,
                total_pests INTEGER NOT NULL,
                counts_json TEXT NOT NULL,
                report_summary TEXT NOT NULL,
                report_detail TEXT NOT NULL,
                report_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pest_control_methods (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pest_name TEXT NOT NULL UNIQUE,
                method_text TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()

        seed_methods = {
            "蚜虫": "优先使用黄色粘虫板监测；轻度发生喷施苦参碱；重度发生时轮换吡虫啉等低抗性药剂。",
            "稻飞虱": "控制氮肥，清除田边杂草；发生初期使用噻嗪酮类药剂并注意轮换。",
            "红蜘蛛": "先进行叶背冲洗与通风降温；药剂可选择阿维菌素与乙螨唑交替使用。",
            "夜蛾": "安装性诱捕器与杀虫灯；低龄幼虫期优先使用苏云金杆菌（Bt）防治。",
        }
        for pest_name, method_text in seed_methods.items():
            conn.execute(
                """
                INSERT OR IGNORE INTO pest_control_methods (pest_name, method_text, updated_at)
                VALUES (?, ?, ?)
                """,
                (pest_name, method_text, datetime.now().isoformat()),
            )
        conn.commit()


MODEL = load_model()
init_db()


@app.route("/")
def index():
    return render_template("index.html", gpu=get_device(), model_path=MODEL_PATH)


def build_report(result, timestamp: str, source_name: str, annotated_name: str) -> dict:
    counts = {}
    detections = []

    for box in result.boxes:
        cls_id = int(box.cls.item())
        cls_name = result.names[cls_id]
        conf = float(box.conf.item())
        x1, y1, x2, y2 = [round(v, 2) for v in box.xyxy.cpu().numpy().reshape(-1).tolist()]

        counts[cls_name] = counts.get(cls_name, 0) + 1
        detections.append(
            {
                "class": cls_name,
                "confidence": round(conf, 4),
                "bbox_xyxy": [x1, y1, x2, y2],
                "width": round(x2 - x1, 2),
                "height": round(y2 - y1, 2),
            }
        )

    total_pests = sum(counts.values())
    summary_rows = [
        {
            "timestamp": timestamp,
            "source_image": source_name,
            "annotated_image": annotated_name,
            "model": MODEL_PATH,
            "device": get_device(),
            "class": cls,
            "count": count,
        }
        for cls, count in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    ]

    if not summary_rows:
        summary_rows = [
            {
                "timestamp": timestamp,
                "source_image": source_name,
                "annotated_image": annotated_name,
                "model": MODEL_PATH,
                "device": get_device(),
                "class": "NONE",
                "count": 0,
            }
        ]

    summary_csv = REPORT_DIR / f"summary_{timestamp}.csv"
    detail_csv = REPORT_DIR / f"details_{timestamp}.csv"
    json_report = REPORT_DIR / f"report_{timestamp}.json"

    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(detections).to_csv(detail_csv, index=False, encoding="utf-8-sig")

    report_payload = {
        "timestamp": timestamp,
        "source_image": source_name,
        "annotated_image": annotated_name,
        "model": MODEL_PATH,
        "device": get_device(),
        "total_pests": total_pests,
        "counts": counts,
        "detections": detections,
        "summary_csv": summary_csv.name,
        "detail_csv": detail_csv.name,
    }
    json_report.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "counts": counts,
        "detections": detections,
        "total_pests": total_pests,
        "summary_csv": summary_csv.name,
        "detail_csv": detail_csv.name,
        "json_report": json_report.name,
    }


def save_history(timestamp: str, source_name: str, annotated_name: str, report: dict):
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO detect_history (
                created_at, source_image, annotated_image, total_pests,
                counts_json, report_summary, report_detail, report_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                source_name,
                annotated_name,
                report["total_pests"],
                json.dumps(report["counts"], ensure_ascii=False),
                report["summary_csv"],
                report["detail_csv"],
                report["json_report"],
            ),
        )
        conn.commit()


@app.route("/api/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "未收到图片文件"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "文件名为空"}), 400

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{timestamp}_{Path(file.filename).name}"
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

    report = build_report(result, timestamp, filename, annotated_name)
    save_history(timestamp, filename, annotated_name, report)

    return jsonify(
        {
            "device": get_device(),
            "input_image": url_for("uploaded_file", filename=filename),
            "annotated_image": url_for("annotated_file", filename=annotated_name),
            "total_pests": report["total_pests"],
            "counts": report["counts"],
            "detections": report["detections"],
            "report_files": {
                "summary_csv": url_for("report_file", filename=report["summary_csv"]),
                "detail_csv": url_for("report_file", filename=report["detail_csv"]),
                "json": url_for("report_file", filename=report["json_report"]),
            },
        }
    )


@app.route("/api/history", methods=["GET"])
def get_history():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM detect_history ORDER BY id DESC LIMIT 200"
        ).fetchall()

    payload = []
    for row in rows:
        payload.append(
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "source_image": url_for("uploaded_file", filename=row["source_image"]),
                "annotated_image": url_for("annotated_file", filename=row["annotated_image"]),
                "total_pests": row["total_pests"],
                "counts": json.loads(row["counts_json"]),
                "report_files": {
                    "summary_csv": url_for("report_file", filename=row["report_summary"]),
                    "detail_csv": url_for("report_file", filename=row["report_detail"]),
                    "json": url_for("report_file", filename=row["report_json"]),
                },
            }
        )
    return jsonify(payload)


@app.route("/api/history/<int:record_id>", methods=["DELETE"])
def delete_history(record_id: int):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM detect_history WHERE id=?", (record_id,)).fetchone()
        if row is None:
            return jsonify({"error": "记录不存在"}), 404

        conn.execute("DELETE FROM detect_history WHERE id=?", (record_id,))
        conn.commit()

    for folder, fname in [
        (UPLOAD_DIR, row["source_image"]),
        (ANNOTATED_DIR, row["annotated_image"]),
        (REPORT_DIR, row["report_summary"]),
        (REPORT_DIR, row["report_detail"]),
        (REPORT_DIR, row["report_json"]),
    ]:
        path = folder / fname
        if path.exists():
            path.unlink()

    return jsonify({"message": "历史记录已删除", "id": record_id})


@app.route("/api/control-methods", methods=["GET"])
def get_control_methods():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, pest_name, method_text, updated_at FROM pest_control_methods ORDER BY pest_name"
        ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/control-methods", methods=["POST"])
def upsert_control_method():
    data = request.get_json(silent=True) or {}
    pest_name = (data.get("pest_name") or "").strip()
    method_text = (data.get("method_text") or "").strip()

    if not pest_name or not method_text:
        return jsonify({"error": "pest_name 与 method_text 均不能为空"}), 400

    now = datetime.now().isoformat()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO pest_control_methods (pest_name, method_text, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(pest_name) DO UPDATE SET
            method_text=excluded.method_text,
            updated_at=excluded.updated_at
            """,
            (pest_name, method_text, now),
        )
        conn.commit()

    return jsonify({"message": "防治方法已保存", "pest_name": pest_name, "updated_at": now})


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

