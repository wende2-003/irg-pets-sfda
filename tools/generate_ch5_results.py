#!/usr/bin/env python
import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class EvalPoint:
    epoch: int
    model: str
    ap: float
    ap50: float
    ap75: float


def parse_log(log_path: Path, year: int = 2026):
    ts_re = re.compile(r"^\[(\d+/\d+ \d+:\d+:\d+)\]")
    epoch_re = re.compile(
        r"\[(\d+/\d+ \d+:\d+:\d+)\] detectron2 INFO: \[EPOCH (\d+)\]\[(STUDENT|TEACHER)\] Evaluation start"
    )
    final_re = re.compile(
        r"\[(\d+/\d+ \d+:\d+:\d+)\] detectron2 INFO: \[FINAL\]\[(STUDENT|TEACHER)\] Evaluation start"
    )
    metric_re = re.compile(r"copypaste: ([0-9.]+),([0-9.]+),([0-9.]+)")

    points = {"student": [], "teacher": []}
    final = {}
    first_ts = None
    last_ts = None
    pending = None

    for line in log_path.read_text(errors="ignore").splitlines():
        ts_m = ts_re.match(line)
        if ts_m:
            ts = datetime.strptime(f"{year}/{ts_m.group(1)}", "%Y/%m/%d %H:%M:%S")
            if first_ts is None:
                first_ts = ts
            last_ts = ts

        e_m = epoch_re.search(line)
        if e_m:
            pending = ("epoch", int(e_m.group(2)), e_m.group(3).lower())
            continue

        f_m = final_re.search(line)
        if f_m:
            pending = ("final", None, f_m.group(2).lower())
            continue

        m_m = metric_re.search(line)
        if not m_m or pending is None:
            continue

        ap, ap50, ap75 = map(float, m_m.groups())
        tag, epoch, model = pending
        pending = None
        if tag == "epoch":
            points[model].append(EvalPoint(epoch, model, ap, ap50, ap75))
        else:
            final[model] = {"ap": ap, "ap50": ap50, "ap75": ap75}

    for k in ("student", "teacher"):
        points[k].sort(key=lambda x: x.epoch)

    runtime_hours = None
    if first_ts and last_ts:
        if last_ts < first_ts:
            last_ts = last_ts + timedelta(days=365)
        runtime_hours = (last_ts - first_ts).total_seconds() / 3600.0

    return {
        "log_path": str(log_path),
        "points": points,
        "final": final,
        "runtime_hours": runtime_hours,
    }


def _extract_series(points, model, metric):
    model_points = points[model]
    xs = [p.epoch for p in model_points]
    ys = [getattr(p, metric) for p in model_points]
    return np.array(xs), np.array(ys, dtype=float)


def plot_metric_curves(run2, run5, metric_name, y_label, out_file):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    colors = {"run2": "#1f77b4", "run5": "#ff7f0e"}

    for run_key, run_data in [("run2", run2), ("run5", run5)]:
        for model, style in [("student", "--"), ("teacher", "-")]:
            xs, ys = _extract_series(run_data["points"], model, metric_name)
            ax.plot(
                xs,
                ys,
                linestyle=style,
                marker="o",
                color=colors[run_key],
                alpha=0.75,
                linewidth=1.8 if model == "teacher" else 1.2,
                label=f"{run_key}-{model}",
            )

    xs = np.array([p.epoch for p in run2["points"]["teacher"]], dtype=float)
    y2 = np.array([getattr(p, metric_name) for p in run2["points"]["teacher"]], dtype=float)
    y5 = np.array([getattr(p, metric_name) for p in run5["points"]["teacher"]], dtype=float)
    mean_y = (y2 + y5) / 2.0
    ax.plot(xs, mean_y, color="#2ca02c", linewidth=2.6, marker="s", label=f"mean-teacher-{y_label}")

    ax.set_xlabel("epoch")
    ax.set_ylabel(y_label)
    ax.set_title(f"{y_label} curves across two runs")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


def plot_teacher_student_gap(run2, run5, out_file):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)

    for run_name, run in [("run2", run2), ("run5", run5)]:
        xs_s, ys_s = _extract_series(run["points"], "student", "ap50")
        xs_t, ys_t = _extract_series(run["points"], "teacher", "ap50")
        if not np.array_equal(xs_s, xs_t):
            raise RuntimeError(f"epoch mismatch in {run_name}")
        gap = ys_t - ys_s
        ax.plot(xs_t, gap, marker="o", linewidth=1.8, label=f"{run_name} teacher-student gap")

    gap_mean = (
        (np.array([p.ap50 for p in run2["points"]["teacher"]]) - np.array([p.ap50 for p in run2["points"]["student"]]))
        + (np.array([p.ap50 for p in run5["points"]["teacher"]]) - np.array([p.ap50 for p in run5["points"]["student"]]))
    ) / 2.0
    epochs = np.array([p.epoch for p in run2["points"]["teacher"]])
    ax.plot(epochs, gap_mean, color="black", linewidth=2.4, marker="s", label="mean gap")

    ax.set_xlabel("epoch")
    ax.set_ylabel("AP50 gap")
    ax.set_title("Teacher-Student AP50 gap")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


def build_summary(run2, run5):
    def _f(run, model, metric):
        return run["final"][model][metric]

    avg_teacher_ap = (_f(run2, "teacher", "ap") + _f(run5, "teacher", "ap")) / 2.0
    avg_teacher_ap50 = (_f(run2, "teacher", "ap50") + _f(run5, "teacher", "ap50")) / 2.0
    avg_student_ap = (_f(run2, "student", "ap") + _f(run5, "student", "ap")) / 2.0
    avg_student_ap50 = (_f(run2, "student", "ap50") + _f(run5, "student", "ap50")) / 2.0
    avg_runtime_h = (run2["runtime_hours"] + run5["runtime_hours"]) / 2.0

    return {
        "run2_final": run2["final"],
        "run5_final": run5["final"],
        "average_final": {
            "teacher": {"ap": avg_teacher_ap, "ap50": avg_teacher_ap50},
            "student": {"ap": avg_student_ap, "ap50": avg_student_ap50},
            "runtime_hours": avg_runtime_h,
        },
        "run2_runtime_hours": run2["runtime_hours"],
        "run5_runtime_hours": run5["runtime_hours"],
    }


def write_summary_files(summary, out_dir: Path):
    json_path = out_dir / "ch5_summary.json"
    md_path = out_dir / "ch5_summary.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    lines = []
    lines.append("# Chapter 5 Summary (auto-generated)")
    lines.append("")
    lines.append("| Item | run2 | run5 | mean |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        "| Teacher AP | {:.4f} | {:.4f} | {:.4f} |".format(
            summary["run2_final"]["teacher"]["ap"],
            summary["run5_final"]["teacher"]["ap"],
            summary["average_final"]["teacher"]["ap"],
        )
    )
    lines.append(
        "| Teacher AP50 | {:.4f} | {:.4f} | {:.4f} |".format(
            summary["run2_final"]["teacher"]["ap50"],
            summary["run5_final"]["teacher"]["ap50"],
            summary["average_final"]["teacher"]["ap50"],
        )
    )
    lines.append(
        "| Student AP | {:.4f} | {:.4f} | {:.4f} |".format(
            summary["run2_final"]["student"]["ap"],
            summary["run5_final"]["student"]["ap"],
            summary["average_final"]["student"]["ap"],
        )
    )
    lines.append(
        "| Student AP50 | {:.4f} | {:.4f} | {:.4f} |".format(
            summary["run2_final"]["student"]["ap50"],
            summary["run5_final"]["student"]["ap50"],
            summary["average_final"]["student"]["ap50"],
        )
    )
    lines.append(
        "| Runtime (hours) | {:.3f} | {:.3f} | {:.3f} |".format(
            summary["run2_runtime_hours"],
            summary["run5_runtime_hours"],
            summary["average_final"]["runtime_hours"],
        )
    )
    md_path.write_text("\n".join(lines))


def generate_qualitative(
    repo_root: Path,
    out_dir: Path,
    cfg_path: Path,
    source_weight: Path,
    ours_weight: Path,
):
    import cv2
    import torch
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog
    from detectron2.data import transforms as T
    from detectron2.modeling import build_model
    from detectron2.utils.visualizer import Visualizer

    def build_eval_model(weight_path: Path):
        cfg = get_cfg()
        cfg.merge_from_file(str(cfg_path))
        cfg.MODEL.WEIGHTS = str(weight_path)
        cfg.MODEL.DEVICE = "cpu"
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.freeze()
        model = build_model(cfg)
        model.eval()
        DetectionCheckpointer(model).load(str(weight_path))
        metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        return cfg, model, metadata

    def predict_instances(model, cfg, image_bgr):
        h, w = image_bgr.shape[:2]
        aug = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
        image = aug.get_transform(image_bgr).apply_image(image_bgr)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        with torch.no_grad():
            outputs = model([{"image": image, "height": h, "width": w}])[0]["instances"].to("cpu")
        return outputs

    def draw_instances(image_bgr, instances, metadata):
        vis = Visualizer(image_bgr[:, :, ::-1], metadata=metadata, scale=0.8)
        out = vis.draw_instance_predictions(instances)
        return out.get_image()

    scenes = [
        ("fog_low_contrast", "lindau_000026_000019_leftImg8bit_foggy"),
        ("strong_light", "frankfurt_000000_009969_leftImg8bit_foggy"),
        ("occlusion_dense", "frankfurt_000001_017101_leftImg8bit_foggy"),
    ]
    image_dir = Path("/root/autodl-tmp/foggycityscape/VOC2007/JPEGImages")

    cfg_src, model_src, meta_src = build_eval_model(source_weight)
    cfg_ours, model_ours, meta_ours = build_eval_model(ours_weight)

    rows = []
    summary_csv = out_dir / "qualitative_highconf_counts.csv"

    montage = []
    for scene_name, image_id in scenes:
        img_path = image_dir / f"{image_id}.png"
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            raise FileNotFoundError(f"image not found: {img_path}")

        src_inst = predict_instances(model_src, cfg_src, image_bgr)
        ours_inst = predict_instances(model_ours, cfg_ours, image_bgr)

        src_draw = draw_instances(image_bgr, src_inst, meta_src)
        ours_draw = draw_instances(image_bgr, ours_inst, meta_ours)

        cv2.imwrite(str(out_dir / f"{scene_name}_raw.png"), image_bgr)
        cv2.imwrite(str(out_dir / f"{scene_name}_source.png"), src_draw[:, :, ::-1])
        cv2.imwrite(str(out_dir / f"{scene_name}_ours.png"), ours_draw[:, :, ::-1])

        src_scores = src_inst.scores.numpy() if src_inst.has("scores") else np.array([])
        ours_scores = ours_inst.scores.numpy() if ours_inst.has("scores") else np.array([])
        rows.append(
            {
                "scene": scene_name,
                "image_id": image_id,
                "source_boxes_ge_0_5": int((src_scores >= 0.5).sum()),
                "source_boxes_ge_0_9": int((src_scores >= 0.9).sum()),
                "ours_boxes_ge_0_5": int((ours_scores >= 0.5).sum()),
                "ours_boxes_ge_0_9": int((ours_scores >= 0.9).sum()),
            }
        )

        montage.append((scene_name, image_bgr[:, :, ::-1], src_draw, ours_draw))

    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(len(montage), 3, figsize=(15, 12), dpi=160)
    if len(montage) == 1:
        axes = np.array([axes])
    col_titles = ["Input", "Source-only", "PETS-SR+IRG (ours)"]
    for j in range(3):
        axes[0, j].set_title(col_titles[j], fontsize=12)
    for i, (scene_name, raw_rgb, src_rgb, ours_rgb) in enumerate(montage):
        for j, img in enumerate([raw_rgb, src_rgb, ours_rgb]):
            axes[i, j].imshow(img)
            axes[i, j].axis("off")
        axes[i, 0].set_ylabel(scene_name, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "qualitative_source_vs_ours.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate chapter-5 figures and summaries.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoint/chapter5_results"))
    parser.add_argument(
        "--run2-log", type=Path, default=Path("checkpoint/foggy_irg_petsr_2/log.txt")
    )
    parser.add_argument(
        "--run5-log", type=Path, default=Path("checkpoint/foggy_irg_petsr_5/log.txt")
    )
    parser.add_argument("--cfg", type=Path, default=Path("configs/sfda/sfda_foggy.yaml"))
    parser.add_argument("--source-weight", type=Path, default=Path("/root/autodl-tmp/model_final.pth"))
    parser.add_argument(
        "--ours-weight", type=Path, default=Path("checkpoint/foggy_irg_petsr_5/model_final.pth")
    )
    parser.add_argument("--skip-qual", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run2 = parse_log((repo_root / args.run2_log).resolve())
    run5 = parse_log((repo_root / args.run5_log).resolve())

    plot_metric_curves(
        run2,
        run5,
        metric_name="ap50",
        y_label="AP50",
        out_file=out_dir / "curve_ap50_epoch.png",
    )
    plot_metric_curves(
        run2,
        run5,
        metric_name="ap",
        y_label="AP",
        out_file=out_dir / "curve_ap_epoch.png",
    )
    plot_teacher_student_gap(run2, run5, out_dir / "curve_teacher_student_gap_ap50.png")

    summary = build_summary(run2, run5)
    write_summary_files(summary, out_dir)

    if not args.skip_qual:
        generate_qualitative(
            repo_root=repo_root,
            out_dir=out_dir,
            cfg_path=(repo_root / args.cfg).resolve(),
            source_weight=args.source_weight.resolve(),
            ours_weight=(repo_root / args.ours_weight).resolve(),
        )

    print(f"Saved results to: {out_dir}")


if __name__ == "__main__":
    main()
