#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
import sys

# Prefer the local detectron2 package in this repo.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import copy
import time
from collections import defaultdict
import torch.optim as optim
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
    ClipartDetectionEvaluator,
    WatercolorDetectionEvaluator,
    CityscapeDetectionEvaluator,
    FoggyDetectionEvaluator,
    CityscapeCarDetectionEvaluator,
)

from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage

import pdb
import cv2
from pynvml import *
from detectron2.structures.boxes import Boxes, pairwise_iou
from detectron2.structures.instances import Instances
from detectron2.data.detection_utils import convert_image_to_rgb

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

logger = logging.getLogger("detectron2")


def _smooth_series(values, window=51):
    if window <= 1 or len(values) < window:
        return values
    half = window // 2
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - half)
        end = min(len(values), i + half + 1)
        smoothed.append(sum(values[start:end]) / (end - start))
    return smoothed


def plot_training_curves(output_dir, history, keys=None, smooth_window=51):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning("Plotting skipped (matplotlib unavailable): %s", exc)
        return

    if not history:
        return

    os.makedirs(output_dir, exist_ok=True)
    steps = history.get("iter", [])
    ap_steps = history.get("ap_iter", [])
    ap_steps_student = history.get("ap_iter_student", [])
    ap_steps_teacher = history.get("ap_iter_teacher", [])
    skip_keys = {
        "iter",
        "ap_iter",
        "ap_iter_student",
        "ap_iter_teacher",
        "ap50_student",
        "ap50_teacher",
        "ap_student",
        "ap_teacher",
    }
    for key, values in history.items():
        if key in skip_keys:
            continue
        if keys is not None and key not in keys:
            continue
        if not values:
            continue
        if key == "ap50":
            student_vals = history.get("ap50_student", [])
            teacher_vals = history.get("ap50_teacher", history.get("ap50", []))
            if student_vals or teacher_vals:
                plt.figure()
                drawn = False
                if student_vals and ap_steps_student:
                    s_vals = _smooth_series(student_vals, smooth_window)
                    plt.plot(
                        ap_steps_student[: len(s_vals)],
                        s_vals,
                        color="tab:blue",
                        marker="o",
                        label="Student AP50",
                    )
                    s_x = ap_steps_student[min(len(ap_steps_student), len(s_vals)) - 1]
                    s_y = s_vals[-1]
                    plt.annotate(
                        f"S {s_y:.2f}",
                        xy=(s_x, s_y),
                        xytext=(4, 6),
                        textcoords="offset points",
                        color="tab:blue",
                        fontsize=8,
                    )
                    drawn = True
                teacher_x = ap_steps_teacher if ap_steps_teacher else ap_steps
                if teacher_vals and teacher_x:
                    t_vals = _smooth_series(teacher_vals, smooth_window)
                    plt.plot(
                        teacher_x[: len(t_vals)],
                        t_vals,
                        color="tab:orange",
                        marker="s",
                        label="Teacher AP50",
                    )
                    t_x = teacher_x[min(len(teacher_x), len(t_vals)) - 1]
                    t_y = t_vals[-1]
                    plt.annotate(
                        f"T {t_y:.2f}",
                        xy=(t_x, t_y),
                        xytext=(4, -10),
                        textcoords="offset points",
                        color="tab:orange",
                        fontsize=8,
                    )
                    drawn = True
                if not drawn:
                    plt.close()
                    continue
                plt.xlabel("epoch")
                plt.ylabel("ap50")
                plt.title("ap50")
                plt.grid(True, linestyle="--", alpha=0.3)
                plt.legend(loc="best")
                plt.text(
                    0.01,
                    0.98,
                    "Blue: Student  Orange: Teacher",
                    transform=plt.gca().transAxes,
                    va="top",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.8),
                )
                out_path = os.path.join(output_dir, "ap50.png")
                plt.tight_layout()
                plt.savefig(out_path, dpi=150)
                plt.close()
                continue
        plt.figure()
        plot_vals = _smooth_series(values, smooth_window)
        if key in ("ap", "ap50") and ap_steps:
            plt.plot(ap_steps[: len(plot_vals)], plot_vals)
            plt.xlabel("epoch")
        else:
            plt.plot(steps[: len(plot_vals)], plot_vals)
            plt.xlabel("iteration")
        plt.ylabel(key)
        plt.title(key)
        plt.grid(True, linestyle="--", alpha=0.3)
        out_path = os.path.join(output_dir, f"{key}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if evaluator_type == "clipart":
        return ClipartDetectionEvaluator(dataset_name)
    if evaluator_type == "watercolor":
        return WatercolorDetectionEvaluator(dataset_name)
    if evaluator_type == "cityscape":
        return CityscapeDetectionEvaluator(dataset_name)
    if evaluator_type == "foggy":
        return FoggyDetectionEvaluator(dataset_name)
    if evaluator_type == "cityscape_car":
        return CityscapeCarDetectionEvaluator(dataset_name)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)

# =====================================================
# ================== Pseduo-labeling ==================
# =====================================================
def threshold_bbox(proposal_bbox_inst, thres=0.7, proposal_type="roih"):
    if proposal_type == "rpn":
        valid_map = proposal_bbox_inst.objectness_logits > thres

        # create instances containing boxes and gt_classes
        image_shape = proposal_bbox_inst.image_size
        new_proposal_inst = Instances(image_shape)

        # create box
        new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
        new_boxes = Boxes(new_bbox_loc)

        # add boxes to instances
        new_proposal_inst.gt_boxes = new_boxes
        new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
            valid_map
        ]
    elif proposal_type == "roih":
        valid_map = proposal_bbox_inst.scores > thres

        # create instances containing boxes and gt_classes
        image_shape = proposal_bbox_inst.image_size
        new_proposal_inst = Instances(image_shape)

        # create box
        new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
        new_boxes = Boxes(new_bbox_loc)

        # add boxes to instances
        new_proposal_inst.gt_boxes = new_boxes
        new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
        new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

    return new_proposal_inst

def process_pseudo_label(proposals_rpn_k, cur_threshold, proposal_type, psedo_label_method=""):
    list_instances = []
    num_proposal_output = 0.0
    for proposal_bbox_inst in proposals_rpn_k:
        # thresholding
        if psedo_label_method == "thresholding":
            proposal_bbox_inst = threshold_bbox(
                proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
            )
        else:
            raise ValueError("Unkown pseudo label boxes methods")
        num_proposal_output += len(proposal_bbox_inst)
        list_instances.append(proposal_bbox_inst)
    num_proposal_output = num_proposal_output / len(proposals_rpn_k)
    return list_instances, num_proposal_output


def _empty_instances(image_size, device):
    empty = Instances(image_size)
    empty.pred_boxes = Boxes(torch.empty((0, 4), device=device))
    empty.scores = torch.empty((0,), device=device)
    empty.pred_classes = torch.empty((0,), dtype=torch.int64, device=device)
    return empty


def consensus_fusion(static_results, dynamic_results, score_thresh=0.5, iou_thresh=0.5, beta=0.5):
    fused_results = []
    for st, dt in zip(static_results, dynamic_results):
        if len(st) == 0 or len(dt) == 0:
            image_size = st.image_size if len(st) > 0 else dt.image_size
            device = st.pred_boxes.tensor.device if len(st) > 0 else dt.pred_boxes.tensor.device
            fused_results.append(_empty_instances(image_size, device))
            continue

        st = st[st.scores > score_thresh]
        dt = dt[dt.scores > score_thresh]
        if len(st) == 0 or len(dt) == 0:
            image_size = st.image_size if len(st) > 0 else dt.image_size
            device = st.pred_boxes.tensor.device if len(st) > 0 else dt.pred_boxes.tensor.device
            fused_results.append(_empty_instances(image_size, device))
            continue

        st_order = torch.argsort(st.scores, descending=True)
        used_dt = set()
        fused_boxes = []
        fused_scores = []
        fused_classes = []

        for si in st_order:
            st_cls = st.pred_classes[si]
            dt_idxs = torch.nonzero(dt.pred_classes == st_cls).view(-1)
            if dt_idxs.numel() == 0:
                continue

            ious = pairwise_iou(st.pred_boxes[si : si + 1], dt.pred_boxes[dt_idxs]).squeeze(0)
            if used_dt:
                for j, idx in enumerate(dt_idxs):
                    if int(idx) in used_dt:
                        ious[j] = -1

            max_iou, max_j = ious.max(0)
            if max_iou < iou_thresh:
                continue

            dt_idx = dt_idxs[max_j]
            used_dt.add(int(dt_idx))

            st_score = st.scores[si]
            dt_score = dt.scores[dt_idx]
            denom = st_score + dt_score
            if denom <= 0:
                continue

            fused_box = (
                st.pred_boxes.tensor[si] * st_score + dt.pred_boxes.tensor[dt_idx] * dt_score
            ) / denom
            fused_score = beta * st_score + (1.0 - beta) * dt_score

            fused_boxes.append(fused_box)
            fused_scores.append(fused_score)
            fused_classes.append(st_cls)

        if fused_boxes:
            fused_boxes = torch.stack(fused_boxes, dim=0)
            fused_scores = torch.stack(fused_scores)
            fused_classes = torch.stack(fused_classes)
            inst = Instances(st.image_size)
            inst.pred_boxes = Boxes(fused_boxes)
            inst.scores = fused_scores
            inst.pred_classes = fused_classes
        else:
            inst = _empty_instances(st.image_size, st.pred_boxes.tensor.device)

        fused_results.append(inst)
    return fused_results


def exchange_shared_weights(model_a, model_b):
    a_state = model_a.state_dict()
    b_state = model_b.state_dict()
    shared = []
    for k, v in a_state.items():
        if k in b_state and b_state[k].shape == v.shape:
            shared.append(k)
    if not shared:
        return
    a_shared = {k: a_state[k].clone() for k in shared}
    b_shared = {k: b_state[k].clone() for k in shared}
    for k in shared:
        a_state[k] = b_shared[k]
        b_state[k] = a_shared[k]
    model_a.load_state_dict(a_state, strict=False)
    model_b.load_state_dict(b_state, strict=False)
@torch.no_grad()
def update_teacher_model(model_student, model_teacher, keep_rate=0.996):
    if comm.get_world_size() > 1:
        student_model_dict = {
            key[7:]: value for key, value in model_student.state_dict().items()
        }
    else:
        student_model_dict = model_student.state_dict()

    new_teacher_dict = OrderedDict()
    for key, value in model_teacher.state_dict().items():
        if key in student_model_dict.keys():
            new_teacher_dict[key] = (
                student_model_dict[key] *
                (1 - keep_rate) + value * keep_rate
            )
        else:
            raise Exception("{} is not found in student model".format(key))

    return new_teacher_dict

def visualize_proposals(cfg, batched_inputs, proposals, box_size, proposal_dir, metadata):
        from detectron2.utils.visualizer import Visualizer

        for input, prop in zip(batched_inputs, proposals):
            img = input["image_weak"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), None)
            #v_gt = Visualizer(img, None)
            #v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            #anno_img = v_gt.get_image()
            v_pred = Visualizer(img, metadata)
            if proposal_dir == "rpn":
                v_pred = v_pred.overlay_instances( boxes=prop.proposal_boxes[0:int(box_size)].tensor.cpu().numpy())
            if proposal_dir == "roih":
                v_pred = v_pred.draw_instance_predictions(prop)
            vis_img = v_pred.get_image()

            save_path = os.path.join(cfg.OUTPUT_DIR, proposal_dir) 
            save_img_path = os.path.join(cfg.OUTPUT_DIR, proposal_dir, input['file_name'].split('/')[-1]) 
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(save_img_path, vis_img)


def test_sfda(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        cfg.defrost()
        cfg.SOURCE_FREE.TYPE = False
        cfg.freeze()
        test_data_loader = build_detection_test_loader(cfg, dataset_name)
        test_metadata = MetadataCatalog.get(dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, test_data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
            #pdb.set_trace()
            cls_names = test_metadata.get("thing_classes")
            cls_aps = results_i['bbox']['class-AP50']
            for i in range(len(cls_aps)):
                logger.info("AP for {}: {}".format(cls_names[i], cls_aps[i]))
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def train_sfda(cfg, model_student, model_dynamic, model_static=None, resume=False, model_dir=""):
    
    use_pets = cfg.SOURCE_FREE.PETS.ENABLED if hasattr(cfg.SOURCE_FREE, "PETS") else False
    exchange_period = max(1, cfg.SOURCE_FREE.PETS.EXCHANGE_PERIOD) if use_pets else 1
    warmup_epochs = max(0, cfg.SOURCE_FREE.PETS.WARMUP_EPOCHS) if use_pets else 0
    ema_keep = cfg.SOURCE_FREE.PETS.EMA_KEEP_RATE if use_pets else 0.9
    conf_thresh = cfg.SOURCE_FREE.PETS.CONF_THRESH if use_pets else 0.9
    iou_thresh = cfg.SOURCE_FREE.PETS.IOU_THRESH if use_pets else 0.5
    beta = cfg.SOURCE_FREE.PETS.BETA if use_pets else 0.5

    checkpoint = copy.deepcopy(model_dynamic.state_dict())

    model_dynamic.eval()
    model_student.train()
    if use_pets and model_static is not None:
        model_static.eval()

    #optimizer = optim.SGD(model_student.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    optimizer = build_optimizer(cfg, model_student)
    scheduler = build_lr_scheduler(cfg, optimizer)
    checkpointer = DetectionCheckpointer(model_student, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)

    #pdb.set_trace()

    data_loader = build_detection_train_loader(cfg)

    total_epochs = max(1, int(cfg.SOURCE_FREE.PETS.TOTAL_EPOCHS)) if use_pets else 10
    len_data_loader = len(data_loader.dataset.dataset.dataset)
    start_iter = 0
    if use_pets and getattr(cfg.SOURCE_FREE.PETS, "EPOCH_ITERS", 0) > 0:
        max_iter = min(len_data_loader, cfg.SOURCE_FREE.PETS.EPOCH_ITERS)
    else:
        # Allow a quick smoke-run by honoring SOLVER.MAX_ITER if it's smaller.
        if cfg.SOLVER.MAX_ITER > 0 and cfg.SOLVER.MAX_ITER < len_data_loader * total_epochs:
            total_epochs = 1
            max_iter = min(len_data_loader, cfg.SOLVER.MAX_ITER)
        else:
            max_iter = len_data_loader
    max_sf_da_iter = total_epochs * max_iter

    checkpoint_state = checkpointer.resume_or_load(model_dir, resume=resume)
    start_global_iter = checkpoint_state.get("iteration", -1) + 1 if resume else 0
    if start_global_iter > 0:
        epoch_start = (start_global_iter // max_iter) + 1
        start_iter = start_global_iter % max_iter
        logger.info(
            "Resuming from iteration {} (epoch {}, iter {})".format(
                start_global_iter, epoch_start, start_iter
            )
        )
        if model_static is not None:
            model_static.load_state_dict(model_student.state_dict(), strict=False)
        model_dynamic.load_state_dict(model_student.state_dict(), strict=False)
    else:
        epoch_start = 1
        if model_dir:
            DetectionCheckpointer(model_dynamic, save_dir=cfg.OUTPUT_DIR).load(model_dir)
            if model_static is not None:
                DetectionCheckpointer(model_static, save_dir=cfg.OUTPUT_DIR).load(model_dir)
        logger.info("Starting training from iteration {}".format(start_iter))

    periodic_checkpointer = PeriodicCheckpointer(checkpointer, len_data_loader, max_iter=max_sf_da_iter)
    writers = default_writers(cfg.OUTPUT_DIR, max_sf_da_iter) if comm.is_main_process() else []

    if cfg.TEST.EVAL_PERIOD > 0:
        model_dynamic.eval()
        test_sfda(cfg, model_dynamic)

    history = defaultdict(list)
    start_wall_time = time.time()
    iter_times = []
    plot_interval = 500
    with EventStorage(start_global_iter) as storage:
        for epoch in range(epoch_start, total_epochs+1):
            cfg.defrost()
            cfg.SOURCE_FREE.TYPE = True
            cfg.freeze()
            data_loader = build_detection_train_loader(cfg)
            model_dynamic.eval()
            if use_pets and model_static is not None:
                model_static.eval()
            model_student.train()
            for iteration, data in enumerate(data_loader):
                if epoch == epoch_start and iteration < start_iter:
                    continue
                if iteration >= max_iter:
                    break
                iter_start = time.time()
                storage.iter = iteration
                optimizer.zero_grad()

                with torch.no_grad():
                    if use_pets and model_static is not None:
                        _, static_features, static_proposals, static_results = model_static(data, mode="train")
                        _, dynamic_features, dynamic_proposals, dynamic_results = model_dynamic(data, mode="train")
                        consensus_results = consensus_fusion(
                            static_results,
                            dynamic_results,
                            score_thresh=conf_thresh,
                            iou_thresh=iou_thresh,
                            beta=beta,
                        )
                        teacher_pseudo_results, _ = process_pseudo_label(
                            consensus_results, 0.0, "roih", "thresholding"
                        )
                    else:
                        _, dynamic_features, dynamic_proposals, dynamic_results = model_dynamic(data, mode="train")
                        teacher_pseudo_results, _ = process_pseudo_label(
                            dynamic_results, conf_thresh, "roih", "thresholding"
                        )

                loss_dict = model_student(
                    data,
                    cfg,
                    model_dynamic,
                    dynamic_features,
                    dynamic_proposals,
                    teacher_pseudo_results,
                    mode="train",
                )

                losses = sum(loss_dict.values())
                assert torch.isfinite(losses).all(), loss_dict

                losses.backward()
                optimizer.step()
                if use_pets:
                    new_dynamic_dict = update_teacher_model(model_student, model_dynamic, keep_rate=ema_keep)
                    model_dynamic.load_state_dict(new_dynamic_dict)
                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)

                global_iter = (epoch - 1) * max_iter + iteration
                history["iter"].append(global_iter)
                history["loss_total"].append(losses.item())
                for k, v in loss_dict.items():
                    history[k].append(v.item())

                iter_times.append(time.time() - iter_start)
                if len(iter_times) > 200:
                    iter_times.pop(0)

                if iteration - start_iter > 5 and ((iteration + 1) % 50 == 0 or iteration == max_iter - 1):
                    avg_iter = sum(iter_times) / max(1, len(iter_times))
                    remaining = (max_sf_da_iter - global_iter - 1)
                    eta_sec = int(remaining * avg_iter)
                    eta_h = eta_sec // 3600
                    eta_m = (eta_sec % 3600) // 60
                    print(
                        "epoch: ",
                        epoch,
                        f"iter: {iteration + 1}/{max_iter}",
                        f"global: {global_iter + 1}/{max_sf_da_iter}",
                        "lr:",
                        optimizer.param_groups[0]["lr"],
                        "ETA:",
                        f"{eta_h:02d}h{eta_m:02d}m",
                        ''.join(['{0}: {1}, '.format(k, v.item()) for k,v in loss_dict.items()])
                    )

                if comm.is_main_process() and plot_interval > 0 and (iteration + 1) % plot_interval == 0:
                    plot_training_curves(cfg.OUTPUT_DIR, history, keys=["loss_total", "ap50", "ap"])


                periodic_checkpointer.step(global_iter)

            if not use_pets:
                new_dynamic_dict = update_teacher_model(model_student, model_dynamic, keep_rate=ema_keep)
                model_dynamic.load_state_dict(new_dynamic_dict)
            else:
                if epoch > warmup_epochs and epoch % exchange_period == 0 and model_static is not None:
                    exchange_shared_weights(model_student, model_static)

            if cfg.TEST.EVAL_PERIOD > 0:
                model_student.eval()
                logger.info("[EPOCH %d][STUDENT] Evaluation start", epoch)
                student_results = test_sfda(cfg, model_student)
                if isinstance(student_results, dict) and "bbox" in student_results:
                    bbox_res = student_results["bbox"]
                    if "AP50" in bbox_res:
                        history["ap50_student"].append(bbox_res["AP50"])
                        history["ap_iter_student"].append(epoch)
                    elif "AP" in bbox_res:
                        history["ap_student"].append(bbox_res["AP"])
                        history["ap_iter_student"].append(epoch)

                model_dynamic.eval()
                logger.info("[EPOCH %d][TEACHER] Evaluation start", epoch)
                teacher_results = test_sfda(cfg, model_dynamic)
                if isinstance(teacher_results, dict) and "bbox" in teacher_results:
                    bbox_res = teacher_results["bbox"]
                    if "AP50" in bbox_res:
                        history["ap50_teacher"].append(bbox_res["AP50"])
                        history["ap_iter_teacher"].append(epoch)
                        history["ap50"].append(bbox_res["AP50"])
                        history["ap_iter"].append(epoch)
                    elif "AP" in bbox_res:
                        history["ap_teacher"].append(bbox_res["AP"])
                        history["ap_iter_teacher"].append(epoch)
                        history["ap"].append(bbox_res["AP"])
                        history["ap_iter"].append(epoch)

                torch.save(model_dynamic.state_dict(), cfg.OUTPUT_DIR + "/model_teacher_{}.pth".format(epoch))
                torch.save(model_student.state_dict(), cfg.OUTPUT_DIR + "/model_student_{}.pth".format(epoch))
    
    if cfg.TEST.EVAL_PERIOD > 0:
        model_student.eval()
        logger.info("[FINAL][STUDENT] Evaluation start")
        test_sfda(cfg, model_student)

        model_dynamic.eval()
        logger.info("[FINAL][TEACHER] Evaluation start")
        test_sfda(cfg, model_dynamic)

    if comm.is_main_process():
        plot_training_curves(cfg.OUTPUT_DIR, history)



def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):

    cfg = setup(args)

    model_student = build_model(cfg)
    cfg_teacher = cfg.clone()
    cfg_teacher.defrost()
    cfg_teacher.MODEL.META_ARCHITECTURE = "teacher_sfda_RCNN"
    cfg_teacher.freeze()
    model_dynamic = build_model(cfg_teacher)
    model_static = build_model(cfg_teacher)
    logger.info("Model:\n{}".format(model_student))

    logger.info("Models built. Starting training...")
    return train_sfda(
        cfg,
        model_student,
        model_dynamic,
        model_static,
        resume=args.resume,
        model_dir=args.model_dir,
    )


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument(
        "--model_dir",
        dest="model_dir",
        help="Alias for --model-dir",
    )
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
