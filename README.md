# 基于改进 YOLOv8m 的害虫检测识别系统

## 功能
- 害虫检测与定位（输出带边界框图像）。
- 按类别统计害虫数量。
- 导出 CSV 检测报告。
- Flask + HTML/CSS/JavaScript 可视化系统。
- 训练默认优先使用 GPU（`cuda:0`）。

## 项目结构
- `scripts/preprocess.py`：数据预处理，自动生成 YOLO 标签，并保存带标识框图像。
- `models/yolov8m_pest_attention.yaml`：改进 YOLOv8m（注意力机制 + 增强特征金字塔）。
- `scripts/train.py`：训练脚本。
- `flask_app/app.py`：后端接口与推理。
- `flask_app/templates`、`flask_app/static`：前端页面。

## 快速开始
```bash
pip install -r requirements.txt
python scripts/preprocess.py
python scripts/train.py
python flask_app/app.py
```

浏览器打开：`http://127.0.0.1:5000`

## 说明
- 若存在 CUDA，训练和推理自动使用 GPU。
- 如需切换模型权重：
```bash
PEST_MODEL_PATH=outputs/train/yolov8m_pest_attention/weights/best.pt python flask_app/app.py
```
