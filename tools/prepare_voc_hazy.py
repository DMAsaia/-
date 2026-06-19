import argparse
import math
import random
import shutil
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import yaml
from tqdm import tqdm


VOC_NAMES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

URLS = {
    "VOCtrainval_06-Nov-2007.zip": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VOCtrainval_06-Nov-2007.zip",
    "VOCtest_06-Nov-2007.zip": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VOCtest_06-Nov-2007.zip",
}


def download_file(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        print(f"exists: {dst.name}")
        return

    print(f"downloading: {url}")
    with urllib.request.urlopen(url) as response, dst.open("wb") as f:
        total = int(response.headers.get("Content-Length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, desc=dst.name) as bar:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))


def extract_zip(src: Path, dst: Path) -> None:
    marker = dst / f".{src.stem}.done"
    if marker.exists():
        print(f"extracted: {src.name}")
        return

    print(f"extracting: {src.name}")
    with ZipFile(src) as zf:
        zf.extractall(dst)
    marker.write_text("ok", encoding="utf-8")


def convert_box(size, box):
    dw, dh = 1.0 / size[0], 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh


def get_label_lines(voc_root: Path, year: str, image_id: str) -> list[str]:
    xml_path = voc_root / f"VOC{year}" / "Annotations" / f"{image_id}.xml"
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    lines = []
    for obj in root.iter("object"):
        cls = obj.find("name").text
        difficult = int(obj.find("difficult").text)
        if cls not in VOC_NAMES or difficult == 1:
            continue
        xmlbox = obj.find("bndbox")
        box = [float(xmlbox.find(x).text) for x in ("xmin", "xmax", "ymin", "ymax")]
        xywh = convert_box((width, height), box)
        lines.append(" ".join(str(v) for v in (VOC_NAMES.index(cls), *xywh)))

    return lines


def has_valid_label(voc_root: Path, year: str, image_id: str) -> bool:
    return bool(get_label_lines(voc_root, year, image_id))


def convert_label(voc_root: Path, year: str, image_id: str, out_file: Path) -> None:
    lines = get_label_lines(voc_root, year, image_id)
    out_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def add_haze(image: np.ndarray, beta: float = 1.35, atmosphere: float = 0.82) -> np.ndarray:
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    center_x, center_y = w * 0.5, h * 0.58
    distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    depth = distance / (math.sqrt(center_x**2 + center_y**2) + 1e-6)
    transmission = np.exp(-beta * depth).astype(np.float32)
    transmission = cv2.GaussianBlur(transmission, (0, 0), sigmaX=25, sigmaY=25)
    transmission = transmission[..., None]

    img = image.astype(np.float32) / 255.0
    hazy = img * transmission + atmosphere * (1.0 - transmission)
    return np.clip(hazy * 255.0, 0, 255).astype(np.uint8)


def beta_tag(beta: float) -> str:
    return f"beta{beta:.2f}"


def build_split(
    voc_root: Path,
    out_root: Path,
    year: str,
    split: str,
    image_split: str,
    betas: list[float],
    suffix_beta: bool,
) -> dict[float, list[str]]:
    ids_file = voc_root / f"VOC{year}" / "ImageSets" / "Main" / f"{image_split}.txt"
    image_ids = ids_file.read_text(encoding="utf-8").strip().split()
    image_ids = [image_id for image_id in image_ids if has_valid_label(voc_root, year, image_id)]

    img_out = out_root / "images" / split
    clean_out = out_root / "clean" / split
    label_out = out_root / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    clean_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    split_items: dict[float, list[str]] = {beta: [] for beta in betas}
    for beta in betas:
        for image_id in tqdm(image_ids, desc=f"{split} {beta_tag(beta)}"):
            stem = f"{image_id}_{beta_tag(beta)}" if suffix_beta else image_id
            src_img = voc_root / f"VOC{year}" / "JPEGImages" / f"{image_id}.jpg"
            dst_img = img_out / f"{stem}.jpg"
            dst_clean = clean_out / f"{stem}.jpg"
            dst_label = label_out / f"{stem}.txt"

            if not dst_img.exists():
                image = cv2.imread(str(src_img))
                if image is None:
                    raise FileNotFoundError(src_img)
                hazy = add_haze(image, beta=beta)
                cv2.imwrite(str(dst_img), hazy, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            if not dst_clean.exists():
                shutil.copy2(src_img, dst_clean)

            if not dst_label.exists():
                convert_label(voc_root, year, image_id, dst_label)

            split_items[beta].append(f"images/{split}/{stem}.jpg")

    return split_items


def write_subset_files(out_root: Path, train_items: dict[float, list[str]], ratios: list[int], seed: int) -> None:
    for ratio in ratios:
        merged: list[str] = []
        for beta, items in train_items.items():
            rng = random.Random(seed + int(beta * 100))
            sampled = items[:]
            rng.shuffle(sampled)
            sample_count = max(1, int(len(sampled) * ratio / 100))
            sampled = sorted(sampled[:sample_count])

            beta_file = out_root / f"train_{beta_tag(beta)}_subset{ratio}.txt"
            beta_file.write_text("\n".join(sampled) + "\n", encoding="utf-8")
            merged.extend(sampled)

        merged = sorted(merged)
        (out_root / f"train_5beta_subset{ratio}.txt").write_text("\n".join(merged) + "\n", encoding="utf-8")


def write_yaml(out_root: Path, train: str = "images/train", suffix: str = "") -> None:
    data = {
        "path": ".",
        "train": train,
        "val": "images/val",
        "test": "images/test",
        "clean": "clean",
        "names": {i: name for i, name in enumerate(VOC_NAMES)},
    }
    yaml_name = f"VOC_hazy{suffix}.yaml"
    with (out_root / yaml_name).open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare VOC2007_hazy in YOLO detection format.")
    parser.add_argument("--root", default="datasets", help="Dataset parent directory.")
    parser.add_argument("--out-dir", default=None, help="Output directory name under --root.")
    parser.add_argument("--beta", type=float, default=1.35, help="Haze density. Larger means heavier fog.")
    parser.add_argument("--betas", nargs="+", type=float, default=None, help="Multiple haze densities for 5-beta data.")
    parser.add_argument("--subset-ratios", nargs="+", type=int, default=[30], help="Layered train subset ratios.")
    parser.add_argument("--subset-seed", type=int, default=42, help="Random seed for deterministic subset sampling.")
    parser.add_argument("--keep-raw-zips", action="store_true", help="Keep downloaded zip files after extraction.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    raw = root / "_raw_voc2007"
    betas = args.betas if args.betas else [args.beta]
    suffix_beta = args.betas is not None or len(betas) > 1
    out_dir = args.out_dir or ("VOC_hazy_5beta" if suffix_beta else "VOC_hazy")
    out_root = root / out_dir
    raw.mkdir(parents=True, exist_ok=True)

    for name, url in URLS.items():
        zip_path = raw / name
        download_file(url, zip_path)
        extract_zip(zip_path, raw)
        if not args.keep_raw_zips:
            zip_path.unlink(missing_ok=True)

    voc_root = raw / "VOCdevkit"
    train_items = build_split(voc_root, out_root, "2007", "train", "train", betas, suffix_beta)
    build_split(voc_root, out_root, "2007", "val", "val", betas, suffix_beta)
    build_split(voc_root, out_root, "2007", "test", "test", betas, suffix_beta)

    if suffix_beta:
        write_subset_files(out_root, train_items, args.subset_ratios, args.subset_seed)
        first_ratio = args.subset_ratios[0]
        write_yaml(out_root, train=f"train_5beta_subset{first_ratio}.txt")
        for ratio in args.subset_ratios:
            write_yaml(out_root, train=f"train_5beta_subset{ratio}.txt", suffix=f"_subset{ratio}")
    else:
        write_yaml(out_root)

    print(f"\nDone: {out_root}")
    print(f"YAML: {out_root / 'VOC_hazy.yaml'}")
    if not args.keep_raw_zips:
        print("Zip files removed after extraction to save disk space.")


if __name__ == "__main__":
    main()
