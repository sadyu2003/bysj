"""使用改进 YOLOv8m 模型训练害虫检测模型（GPU 优先）。"""

import torch
from ultralytics import YOLO

MODEL_CFG = "models/yolov8m_pest_attention.yaml"
DATA_CFG = "datasets/pest_det/pests.yaml"


def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"训练设备: {'cuda:0' if device == 0 else 'cpu'}")

    model = YOLO(MODEL_CFG)
    model.train(
        data=DATA_CFG,
        epochs=100,
        imgsz=960,
        batch=8,
        device=device,
        workers=4,
        optimizer="AdamW",
        cos_lr=True,
        close_mosaic=10,
        pretrained="yolov8m.pt",
        project="outputs/train",
        name="yolov8m_pest_attention",
    )


if __name__ == "__main__":
    main()
