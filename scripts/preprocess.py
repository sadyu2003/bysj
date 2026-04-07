"""将分类目录数据转换为 YOLO 检测数据结构，并自动生成标识框图片与标签。"""

from pathlib import Path
import random
import cv2

SRC_DIR = Path("pests")
OUT_DIR = Path("datasets/pest_det")
SPLIT = 0.9
SEED = 42


def ensure_dirs(base: Path):
    for split in ["train", "val"]:
        (base / "images" / split).mkdir(parents=True, exist_ok=True)
        (base / "labels" / split).mkdir(parents=True, exist_ok=True)


def estimate_bbox(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0, 0, w - 1, h - 1
    c = max(cnts, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(c)
    return x, y, x + bw, y + bh


def to_yolo(x1, y1, x2, y2, w, h):
    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return cx, cy, bw, bh


def main():
    random.seed(SEED)
    ensure_dirs(OUT_DIR)

    classes = sorted([p.name for p in SRC_DIR.iterdir() if p.is_dir()])
    class_to_id = {c: i for i, c in enumerate(classes)}

    yaml_text = (
        f"path: {OUT_DIR.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names: {classes}\n"
    )
    (OUT_DIR / "pests.yaml").write_text(yaml_text, encoding="utf-8")

    for cls in classes:
        images = list((SRC_DIR / cls).glob("*.jpg")) + list((SRC_DIR / cls).glob("*.png"))
        random.shuffle(images)
        split_idx = int(len(images) * SPLIT)
        train_imgs, val_imgs = images[:split_idx], images[split_idx:]

        for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs)]:
            for i, img_path in enumerate(split_imgs):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h, w = img.shape[:2]
                x1, y1, x2, y2 = estimate_bbox(img)
                cx, cy, bw, bh = to_yolo(x1, y1, x2, y2, w, h)
                stem = f"{cls}_{i:05d}"

                out_img = OUT_DIR / "images" / split_name / f"{stem}.jpg"
                out_lbl = OUT_DIR / "labels" / split_name / f"{stem}.txt"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imwrite(str(out_img), img)
                out_lbl.write_text(
                    f"{class_to_id[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n",
                    encoding="utf-8",
                )

    print("预处理完成，输出目录:", OUT_DIR)


if __name__ == "__main__":
    main()
