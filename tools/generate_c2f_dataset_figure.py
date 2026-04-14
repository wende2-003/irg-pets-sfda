#!/usr/bin/env python
import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle
from PIL import Image

CLASS_COLORS = {
    "person": "#e63946",
    "rider": "#f4a261",
    "car": "#1d3557",
    "truck": "#457b9d",
    "bus": "#2a9d8f",
    "train": "#6a4c93",
    "motorcycle": "#ef476f",
    "bicycle": "#118ab2",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a paper-ready C2F dataset figure from paired Cityscapes/Foggy Cityscapes data."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/root/autodl-tmp/cityscape/VOC2007"),
        help="VOC2007 root of Cityscapes source domain",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("/root/autodl-tmp/foggycityscape/VOC2007"),
        help="VOC2007 root of Foggy Cityscapes target domain",
    )
    parser.add_argument("--split", type=str, default="test", help="Image split: trainval or test")
    parser.add_argument("--num-rows", type=int, default=3, help="Number of paired samples to show")
    parser.add_argument("--max-boxes", type=int, default=10, help="Max number of boxes per image")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dpi", type=int, default=320, help="Output DPI")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("imgs/c2f_cityscapes_foggy_benchmark.png"),
        help="Output figure path",
    )
    return parser.parse_args()


def load_split_ids(split_path: Path):
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    return [line.strip() for line in split_path.read_text().splitlines() if line.strip()]


def parse_voc_objects(xml_path: Path):
    if not xml_path.exists():
        return []

    root = ET.parse(xml_path).getroot()
    objects = []
    for node in root.findall("object"):
        cls_name = node.findtext("name", default="unknown").strip()
        bbox = node.find("bndbox")
        if bbox is None:
            continue

        try:
            xmin = float(bbox.findtext("xmin", default="0"))
            ymin = float(bbox.findtext("ymin", default="0"))
            xmax = float(bbox.findtext("xmax", default="0"))
            ymax = float(bbox.findtext("ymax", default="0"))
        except ValueError:
            continue

        if xmax <= xmin or ymax <= ymin:
            continue

        area = (xmax - xmin) * (ymax - ymin)
        objects.append(
            {
                "name": cls_name,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "area": area,
            }
        )

    return objects


def choose_boxes(objects, img_w, img_h, max_boxes):
    if not objects:
        return []

    sorted_objs = sorted(objects, key=lambda x: x["area"], reverse=True)
    large_thresh = 0.0025 * img_w * img_h
    large_objs = [obj for obj in sorted_objs if obj["area"] >= large_thresh]

    if len(large_objs) >= 5:
        chosen = large_objs[:max_boxes]
    else:
        chosen = sorted_objs[:max_boxes]

    return chosen


def build_candidates(sample_ids, src_root: Path, tgt_root: Path):
    src_img_dir = src_root / "JPEGImages"
    tgt_img_dir = tgt_root / "JPEGImages"
    src_ann_dir = src_root / "Annotations"

    candidates = []
    for sample_id in sample_ids:
        src_img = src_img_dir / f"{sample_id}.png"
        tgt_img = tgt_img_dir / f"{sample_id}_foggy.png"
        src_ann = src_ann_dir / f"{sample_id}.xml"
        if not (src_img.exists() and tgt_img.exists() and src_ann.exists()):
            continue

        objects = parse_voc_objects(src_ann)
        if not objects:
            continue

        classes = {obj["name"] for obj in objects}
        top_areas = sorted((obj["area"] for obj in objects), reverse=True)[:5]
        large_count = sum(1 for a in top_areas if a > 12000)

        score = (
            1.4 * len(classes)
            + 0.03 * len(objects)
            + 1.8 * large_count
            + 0.00004 * sum(top_areas)
        )

        city_name = sample_id.split("_")[0]
        candidates.append(
            {
                "id": sample_id,
                "city": city_name,
                "score": score,
                "src_img": src_img,
                "tgt_img": tgt_img,
                "objects": objects,
            }
        )

    return sorted(candidates, key=lambda x: x["score"], reverse=True)


def pick_representative_samples(candidates, k, seed=42):
    if len(candidates) <= k:
        return candidates

    random.seed(seed)
    top_pool = candidates[: min(len(candidates), 120)]

    selected = []
    used_cities = set()

    random.shuffle(top_pool)
    top_pool = sorted(top_pool, key=lambda x: x["score"], reverse=True)

    for item in top_pool:
        if item["city"] in used_cities:
            continue
        selected.append(item)
        used_cities.add(item["city"])
        if len(selected) == k:
            return selected

    for item in top_pool:
        if item in selected:
            continue
        selected.append(item)
        if len(selected) == k:
            return selected

    return selected


def gather_split_stats(src_root: Path, tgt_root: Path):
    src_trainval = src_root / "ImageSets" / "Main" / "trainval.txt"
    src_test = src_root / "ImageSets" / "Main" / "test.txt"
    tgt_trainval = tgt_root / "ImageSets" / "Main" / "trainval.txt"
    tgt_test = tgt_root / "ImageSets" / "Main" / "test.txt"

    def count_lines(path: Path):
        if not path.exists():
            return 0
        return sum(1 for line in path.read_text().splitlines() if line.strip())

    stats = {
        "src_trainval": count_lines(src_trainval),
        "src_test": count_lines(src_test),
        "tgt_trainval": count_lines(tgt_trainval),
        "tgt_test": count_lines(tgt_test),
    }
    return stats


def draw_figure(samples, stats, output_path: Path, max_boxes=10, dpi=320):
    n_rows = len(samples)
    if n_rows == 0:
        raise RuntimeError("No samples available to draw.")

    fig_h = 2.9 * n_rows + 2.2
    fig, axes = plt.subplots(n_rows, 2, figsize=(15.5, fig_h))

    if n_rows == 1:
        axes = np.array([axes])

    fig.suptitle(
        "Cityscapes to Foggy Cityscapes (C2F): Representative Cross-Domain Pairs",
        fontsize=18,
        fontweight="bold",
        y=0.985,
    )
    fig.text(0.25, 0.943, "Source Domain: Cityscapes (clear weather)", ha="center", fontsize=12, color="#1d3557")
    fig.text(0.75, 0.943, "Target Domain: Foggy Cityscapes (adverse weather)", ha="center", fontsize=12, color="#2a9d8f")

    used_classes = set()
    row_centers = []

    for idx, sample in enumerate(samples):
        ax_src = axes[idx, 0]
        ax_tgt = axes[idx, 1]

        src_img = np.array(Image.open(sample["src_img"]).convert("RGB"))
        tgt_img = np.array(Image.open(sample["tgt_img"]).convert("RGB"))

        ax_src.imshow(src_img)
        ax_tgt.imshow(tgt_img)

        img_h, img_w = src_img.shape[:2]
        chosen = choose_boxes(sample["objects"], img_w, img_h, max_boxes=max_boxes)

        for obj in chosen:
            cls_name = obj["name"]
            used_classes.add(cls_name)
            color = CLASS_COLORS.get(cls_name, "#f8f9fa")

            for ax in (ax_src, ax_tgt):
                rect = Rectangle(
                    (obj["xmin"], obj["ymin"]),
                    obj["xmax"] - obj["xmin"],
                    obj["ymax"] - obj["ymin"],
                    linewidth=1.5,
                    edgecolor=color,
                    facecolor="none",
                    alpha=0.92,
                )
                ax.add_patch(rect)

        ax_src.set_title(f"{sample['id']}", fontsize=10)
        ax_tgt.set_title(f"{sample['id']}_foggy", fontsize=10)

        for ax in (ax_src, ax_tgt):
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.3)

        for spine in ax_src.spines.values():
            spine.set_edgecolor("#1d3557")
        for spine in ax_tgt.spines.values():
            spine.set_edgecolor("#2a9d8f")

        left_box = ax_src.get_position()
        row_centers.append((left_box.y0 + left_box.y1) / 2.0)

    for yc in row_centers:
        fig.text(
            0.5,
            yc,
            "C2F domain shift",
            ha="center",
            va="center",
            fontsize=10,
            color="#264653",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#e9f5f2", edgecolor="#2a9d8f", alpha=0.85),
        )

    stats_text = (
        "Paired benchmark split: "
        f"Cityscapes trainval/test = {stats['src_trainval']}/{stats['src_test']}, "
        f"Foggy Cityscapes trainval/test = {stats['tgt_trainval']}/{stats['tgt_test']}"
    )
    fig.text(0.5, 0.055, stats_text, ha="center", fontsize=10)

    legend_classes = [cls for cls in CLASS_COLORS if cls in used_classes]
    if legend_classes:
        handles = [Patch(facecolor="none", edgecolor=CLASS_COLORS[c], linewidth=2, label=c) for c in legend_classes]
        fig.legend(handles=handles, loc="lower center", ncol=min(8, len(handles)), frameon=False, bbox_to_anchor=(0.5, 0.012))

    plt.tight_layout(rect=[0.02, 0.08, 0.98, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    split_path = args.source_root / "ImageSets" / "Main" / f"{args.split}.txt"

    sample_ids = load_split_ids(split_path)
    candidates = build_candidates(sample_ids, args.source_root, args.target_root)
    if not candidates:
        raise RuntimeError("No valid paired samples found. Check dataset paths and split files.")

    samples = pick_representative_samples(candidates, args.num_rows, seed=args.seed)
    stats = gather_split_stats(args.source_root, args.target_root)

    draw_figure(
        samples=samples,
        stats=stats,
        output_path=args.output,
        max_boxes=args.max_boxes,
        dpi=args.dpi,
    )

    print(f"Saved figure: {args.output}")
    print("Selected IDs:")
    for item in samples:
        print(f"  - {item['id']} (score={item['score']:.2f}, city={item['city']})")


if __name__ == "__main__":
    main()
