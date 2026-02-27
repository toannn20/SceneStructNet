import xml.etree.ElementTree as ET
import json
import argparse
import os

def parse_cvat_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    images = []
    annotations =[]
    ann_id = 0

    for image_elem in root.findall("image"):
        img_id = int(image_elem.get("id"))
        file_name = image_elem.get("name")
        width = int(image_elem.get("width"))
        height = int(image_elem.get("height"))

        images.append({
            "id": img_id,
            "file_name": file_name,
            "width": width,
            "height": height,
        })

        for polyline in image_elem.findall("polyline"):
            points_str = polyline.get("points")
            pts = [pt.split(",") for pt in points_str.split(";")]

            if len(pts) < 2:
                continue

            # Process every segment of the polyline
            for i in range(len(pts) - 1):
                x1, y1 = float(pts[i][0]), float(pts[i][1])
                x2, y2 = float(pts[i+1][0]), float(pts[i+1][1])

                # LINEA format:[x1, y1, delta_x, delta_y]
                dx = x2 - x1
                dy = y2 - y1

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 0,
                    "line":[x1, y1, dx, dy],
                    "area": 1,
                })
                ann_id += 1

    # Exact match to the author's categories mapping
    categories =[{"supercategory": "line", "id": "0", "name": "line"}]

    return {"images": images, "annotations": annotations, "categories": categories}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-xml", required=True, help="Path to train CVAT XML")
    parser.add_argument("--val-xml", required=True, help="Path to val CVAT XML")
    parser.add_argument("--test-xml", required=True, help="Path to test CVAT XML")
    parser.add_argument("--output-dir", required=True, help="Output dataset root dir")
    args = parser.parse_args()

    ann_dir = os.path.join(args.output_dir, "annotations")
    os.makedirs(ann_dir, exist_ok=True)

    for xml_path, split in[(args.train_xml, "train"), (args.valid_xml, "val"), (args.test_xml, "test")]:
        data = parse_cvat_xml(xml_path)
        mode = 'lines'
        out_file = os.path.join(ann_dir, f"{mode}_{split}_ann.json")
        with open(out_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(data['images'])} images and {len(data['annotations'])} lines → {out_file}")

    print(f"\nDataset ready at: {args.output_dir}")

if __name__ == "__main__":
    main()