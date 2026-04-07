# 基于改进 YOLOv8m 的害虫检测识别系统

## 功能
- 图像上传后自动检测害虫，返回带边界框（标识框）的结果图。
- 自动统计图片中害虫数量（总数 + 各类别数量）。
- 自动生成检测报告：汇总 CSV、明细 CSV、结构化 JSON。
- 数据管理模块：保存检测结果、查看历史检测记录、删除历史记录。
- 害虫防治方法库：支持查看与新增/更新防治方法。
- Flask + HTML/CSS/JavaScript 可视化系统。
- 训练与推理默认优先使用 GPU（`cuda:0`）。

## 项目结构
- `scripts/preprocess.py`：数据预处理，自动生成 YOLO 标签，并保存带标识框图像。
- `models/yolov8m_pest_attention.yaml`：改进 YOLOv8m（注意力机制 + 增强特征金字塔）。
- `scripts/train.py`：训练脚本。
- `flask_app/app.py`：后端接口、报告生成、历史记录管理、防治方法库。
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
- 系统运行后会在 `data/pest_system.db` 中维护历史记录与防治方法库。
- 如需切换模型权重：
```bash
PEST_MODEL_PATH=outputs/train/yolov8m_pest_attention/weights/best.pt python flask_app/app.py
```
