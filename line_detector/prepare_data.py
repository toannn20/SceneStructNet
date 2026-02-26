import argparse
import json
import math
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_cvat_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    records = []
    for image_elem in root.findall("image"):
        filename = image_elem.get("name")
        width = int(image_elem.get("width"))
        height = int(image_elem.get("height"))
        lines = []
        for polyline in image_elem.findall("polyline"):
            if polyline.get("label") != "vertical_line":
                continue
            pts = polyline.get("points").split(";")
            if len(pts) < 2:
                continue
            p1, p2 = pts[0].split(","), pts[1].split(",")
            x1, y1, x2, y2 = float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])
            if y1 > y2:
                x1, y1, x2, y2 = x2, y2, x1, y1
            angle = math.degrees(math.atan2(abs(x2 - x1), abs(y2 - y1)))
            lines.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                          "angle_from_vertical": round(angle, 2)})
        records.append({"filename": filename, "width": width,
                        "height": height, "lines": lines})
    return records


def split_and_prepare(dataset_root, output_dir="data", images_out_dir="data/images",
                      train_ratio=0.8, seed=42):
    random.seed(seed)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(images_out_dir).mkdir(parents=True, exist_ok=True)

    all_train, all_val = [], []

    for category_dir in sorted(Path(dataset_root).iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        print(f"\n[{category}]")

        xml_files = list(category_dir.rglob("annotations.xml")) or list(category_dir.rglob("*.xml"))
        if not xml_files:
            print("  No XML found, skipping.")
            continue

        records = []
        for xml_file in xml_files:
            records.extend(parse_cvat_xml(xml_file))

        if not records:
            print("  No annotated images found.")
            continue

        image_dir = category_dir / "images" if (category_dir / "images").exists() else category_dir
        random.shuffle(records)
        split_idx = max(1, int(len(records) * train_ratio))
        train_records, val_records = records[:split_idx], records[split_idx:]

        dest = Path(images_out_dir) / category
        dest.mkdir(parents=True, exist_ok=True)

        def copy_and_remap(recs):
            for rec in recs:
                name = Path(rec["filename"]).name
                for src in [image_dir / rec["filename"], image_dir / name, category_dir / rec["filename"]]:
                    if src.exists():
                        shutil.copy2(src, dest / name)
                        rec["filename"] = str(Path(category) / name)
                        break

        copy_and_remap(train_records)
        copy_and_remap(val_records)
        all_train.extend(train_records)
        all_val.extend(val_records)

        print(f"  Train: {len(train_records)} images, {sum(len(r['lines']) for r in train_records)} lines")
        print(f"  Val:   {len(val_records)} images, {sum(len(r['lines']) for r in val_records)} lines")

    for name, data in [("train", all_train), ("val", all_val)]:
        with open(f"{output_dir}/{name}.json", "w") as f:
            json.dump(data, f, indent=2)

    print(f"\nTotal — train: {len(all_train)} | val: {len(all_val)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--images_out_dir", default="data/images")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_and_prepare(args.dataset_root, args.output_dir,
                      args.images_out_dir, args.train_ratio, args.seed)
