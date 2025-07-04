# Evaluates semantic instance task
# Adapted from the CityScapes evaluation: https://github.com/mcordts/cityscapesScripts/tree/master/cityscapesscripts/evaluation
# Input:
#   - path to .txt prediction files
#   - path to .txt ground truth files
#   - output file to write results to
# Each .txt prediction file look like:
#    [(pred0) rel. path to pred. mask over verts as .txt] [(pred0) label id] [(pred0) confidence]
#    [(pred1) rel. path to pred. mask over verts as .txt] [(pred1) label id] [(pred1) confidence]
#    [(pred2) rel. path to pred. mask over verts as .txt] [(pred2) label id] [(pred2) confidence]
#    ...
#
# NOTE: The prediction files must live in the root of the given prediction path.
#       Predicted mask .txt files must live in a subfolder.
#       Additionally, filenames must not contain spaces.
# The relative paths to predicted masks must contain one integer per line,
# where each line corresponds to vertices in the *_vh_clean_2.ply (in that order).
# Non-zero integers indicate part of the predicted instance.
# The label ids specify the class of the corresponding mask.
# Confidence is a float confidence score of the mask.
#
# Note that only the valid classes are used for evaluation,
# i.e., any ground truth label not in the valid label set
# is ignored in the evaluation.
#
# example usage: evaluate_semantic_instance.py --scan_path [path to scan data] --output_file [output file]

# python imports
import math
import os, sys, argparse
import inspect
from copy import deepcopy
from uuid import uuid4

import torch

try:
    import numpy as np
except:
    print("Failed to import numpy package.")
    sys.exit(-1)

from scipy import stats

# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0,parentdir)
import benchmark.util as util
import benchmark.util_3d as util_3d

# parser = argparse.ArgumentParser()
# parser.add_argument('--gt_path', default='', help='path to directory of gt .txt files')
# parser.add_argument('--output_file', default='', help='output file [default: ./semantic_instance_evaluation.txt]')
# opt = parser.parse_args()

# if opt.output_file == '':
#    opt.output_file = os.path.join(os.getcwd(), 'semantic_instance_evaluation.txt')
class UniqueIDs:
    def __init__(self):
        self.uuids = set()

    def get_id(self):
        while True:
            # get random positive integer in int32 range
            uuid = np.random.randint(1, 2 ** 31 - 1)
            if not uuid in self.uuids:
                self.uuids.add(uuid)
                return uuid
unique_ids = UniqueIDs()
# ---------- Label info ---------- #
CLASS_LABELS = [
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
]
VALID_CLASS_IDS = np.array(
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
)
ID_TO_LABEL = {}
LABEL_TO_ID = {}
for i in range(len(VALID_CLASS_IDS)):
    LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
    ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]

# ---------- gt and pred association ---------- #
def match_criteria_M_pred(inst_ref, inst_check, iou_th, gt_first = False):
    overlap = float(inst_ref["intersection"]) / (
        inst_check["vert_count"]
        + inst_ref["vert_count"]
        - inst_ref["intersection"]
    )
    if overlap > iou_th:
        return True
    return False

def match_criteria_MO_pred(inst_ref, inst_check, iou_th, gt_first = False):
    origin_th = 0.25
    is_M = match_criteria_M_pred(inst_ref, inst_check, iou_th, gt_first)
    is_origin = False
    arti_label = inst_ref['label_id']
    if ID_TO_LABEL[arti_label] == 'translation':
        return is_M
    # calculate the distance of pred origin to gt axis
    if gt_first:
        pred_origin = inst_check['origin']
        gt_origin = inst_ref['origin']
        gt_axis = inst_ref['axis']
    else:
        pred_origin = inst_ref['origin']
        gt_axis = inst_check['axis']
        gt_origin = inst_check['origin']
    # calculate the distance of pred origin to gt axis
    pred_origin = pred_origin - gt_origin
    pred_origin = pred_origin - np.dot(pred_origin, gt_axis) * gt_axis / np.linalg.norm(gt_axis)
    is_origin = np.linalg.norm(pred_origin) < origin_th   
    return is_M and is_origin

def match_criteria_MA_pred(inst_ref, inst_check, iou_th, gt_first = False):
    """
    match if iou > iou_th and axis difference < axis_th (degree)
    """
    axis_th = 15
    overlap = float(inst_ref["intersection"]) / (
        inst_check["vert_count"]
        + inst_ref["vert_count"]
        - inst_ref["intersection"]
    )
    is_iou = overlap > iou_th
    axis_ref = inst_ref['axis']
    axis_checked = inst_check['axis']
    deriavation = np.dot(axis_ref, axis_checked) / (np.linalg.norm(axis_ref) * np.linalg.norm(axis_checked))
    deriavation = np.abs(deriavation)
    is_axis = np.arccos(deriavation) < np.deg2rad(axis_th)
    return is_iou and is_axis

def match_criteria_MAO_pred(inst_ref, inst_check, iou_th,  gt_first = False):
    axis_th = 15
    origin_th = 0.25,
    is_MA = match_criteria_MA_pred(inst_ref, inst_check, iou_th, gt_first)
    origin_ref = inst_ref['origin']
    origin_checked = inst_check['origin']
    is_origin = np.linalg.norm(origin_ref - origin_checked) < origin_th
    return is_MA and is_origin

def match_criteria_MAO_pred_standard(inst_ref, inst_check, iou_th, gt_first = False):
    axis_th = 15
    origin_th = 0.25
    is_MA = match_criteria_MA_pred(inst_ref, inst_check, iou_th, gt_first)
    is_origin = False
    arti_label = inst_ref['label_id']
    if ID_TO_LABEL[arti_label] == 'translation':
        return is_MA
    # calculate the distance of pred origin to gt axis
    if gt_first:
        pred_origin = inst_check['origin']
        pred_axis = inst_check['axis']
        gt_origin = inst_ref['origin']
        gt_axis = inst_ref['axis']
    else:
        pred_origin = inst_ref['origin']
        pred_axis = inst_ref['axis']
        gt_axis = inst_check['axis']
        gt_origin = inst_check['origin']
    # calculate the distance of pred origin to gt axis
    diff_pred_gt = pred_origin - gt_origin
    # pred_origin = pred_origin - gt_origin
    # pred_origin = pred_origin - np.dot(pred_origin, gt_axis) * gt_axis / np.linalg.norm(gt_axis)
    diff_proj_gt_axis = diff_pred_gt - np.dot(diff_pred_gt, gt_axis) * gt_axis / np.linalg.norm(gt_axis)
    diff_proj_pred_axis = diff_pred_gt - np.dot(diff_pred_gt, pred_axis) * pred_axis / np.linalg.norm(pred_axis)
    is_origin = (np.linalg.norm(diff_proj_gt_axis) < origin_th) and (np.linalg.norm(diff_proj_pred_axis) < origin_th)
    return is_MA and is_origin

def match_criteria_I(inst_ref, inst_check, iou_th, gt_first = False):
    if 'interaction_mask' not in inst_ref or 'interaction_mask' not in inst_check:
        print("gt_first: ", gt_first)
    
    interaction_mask_ref = inst_ref['interaction_mask']
    interaction_mask_check = inst_check['interaction_mask']
    
    num_ref = np.count_nonzero(interaction_mask_ref)
    num_check = np.count_nonzero(interaction_mask_check)
    num_intersection = np.count_nonzero(np.logical_and(interaction_mask_ref, interaction_mask_check))
    
    union = num_ref + num_check - num_intersection
    if union == 0:
        return False
    
    iou = num_intersection / union
    # if gt_first:
    #     overlap = float(inst_ref["intersection"]) / (
    #         inst_check["vert_count"]
    #         + inst_ref["vert_count"]
    #         - inst_ref["intersection"]
    #     )
        # if overlap > 0.5:
        #     print("num_mov_overlap:{}, num_gt: {}, num_pred: {}, num_intersection: {}, iou: {}".format(inst_ref["intersection"], num_ref, num_check, num_intersection, iou))
    return iou >= iou_th

def match_criteria_I_out(inst_ref, inst_check, iou_th, gt_first = False):

    if gt_first:
        interaction_mask_gt = inst_ref['interaction_mask']
        interaction_mask_pred = inst_check['interaction_out']
    else:
        interaction_mask_gt = inst_check['interaction_mask']
        interaction_mask_pred = inst_ref['interaction_out']
    
    num_ref = np.count_nonzero(interaction_mask_gt)
    num_check = np.count_nonzero(interaction_mask_pred)
    num_intersection = np.count_nonzero(np.logical_and(interaction_mask_gt, interaction_mask_pred))
    
    union = num_ref + num_check - num_intersection
    if union == 0:
        return False
    iou = num_intersection / union
    return iou >= iou_th

def match_criteria_I_vector(inst_ref, inst_check, iou_th, gt_first = False):

    if gt_first:
        interaction_mask_gt = inst_ref['interaction_mask']
        interaction_mask_pred = inst_check['interaction_vector_mask']
    else:
        interaction_mask_gt = inst_check['interaction_mask']
        interaction_mask_pred = inst_ref['interaction_vector_mask']
    
    num_ref = np.count_nonzero(interaction_mask_gt)
    num_check = np.count_nonzero(interaction_mask_pred)
    num_intersection = np.count_nonzero(np.logical_and(interaction_mask_gt, interaction_mask_pred))
    
    union = num_ref + num_check - num_intersection
    if union == 0:
        return False
    
    iou = num_intersection / union
    # if gt_first:
    #     overlap = float(inst_ref["intersection"]) / (
    #         inst_check["vert_count"]
    #         + inst_ref["vert_count"]
    #         - inst_ref["intersection"]
    #     )
        # if overlap > 0.5:
        #     print("num_mov_overlap:{}, num_gt: {}, num_pred: {}, num_intersection: {}, iou: {}".format(inst_ref["intersection"], num_ref, num_check, num_intersection, iou))
    return iou >= iou_th

def match_criteria_I_GT(inst_ref, inst_check, iou_th, gt_first = False):

    if gt_first:
        interaction_mask_gt = inst_ref['interaction_mask']
        interaction_mask_pred = inst_check['interaction_mask'].copy()
        gt_mov_masks = inst_ref['mov_mask']
    else:
        interaction_mask_gt = inst_check['interaction_mask']
        interaction_mask_pred = inst_ref['interaction_mask'].copy()
        gt_mov_masks = inst_check['mov_mask']
    interaction_mask_pred[~gt_mov_masks] = False
    assert interaction_mask_gt[~gt_mov_masks].all() == False
    
    num_ref = np.count_nonzero(interaction_mask_gt)
    num_check = np.count_nonzero(interaction_mask_pred)
    num_intersection = np.count_nonzero(np.logical_and(interaction_mask_gt, interaction_mask_pred))
    
    union = num_ref + num_check - num_intersection
    if union == 0:
        return False
    
    iou = num_intersection / union
    return iou >= iou_th

def match_criteria_I_out_GT(inst_ref, inst_check, iou_th, gt_first = False):
    if gt_first:
        interaction_mask_gt = inst_ref['interaction_mask']
        interaction_mask_pred = inst_check['interaction_out'].copy()
        gt_mov_masks = inst_ref['mov_mask']
    else:
        interaction_mask_gt = inst_check['interaction_mask']
        interaction_mask_pred = inst_ref['interaction_out'].copy()
        gt_mov_masks = inst_check['mov_mask']
    interaction_mask_pred[~gt_mov_masks] = False
    assert interaction_mask_gt[~gt_mov_masks].all() == False
    
    num_ref = np.count_nonzero(interaction_mask_gt)
    num_check = np.count_nonzero(interaction_mask_pred)
    num_intersection = np.count_nonzero(np.logical_and(interaction_mask_gt, interaction_mask_pred))
    
    union = num_ref + num_check - num_intersection
    if union == 0:
        return False
    
    iou = num_intersection / union
    return iou >= iou_th

def match_criteria_MAO_ST_I(inst_ref, inst_check, iou_th, gt_first = False):
    is_MAO_ST = match_criteria_MAO_pred_standard(inst_ref, inst_check, iou_th, gt_first)
    is_I = match_criteria_I(inst_ref, inst_check, iou_th, gt_first)
    return is_MAO_ST and is_I    

# ---------- Evaluation params ---------- #
# overlaps for evaluation
opt = {}
opt["overlaps"] = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
# minimum region size for evaluation [verts]
opt["min_region_sizes"] = np.array([1])  # 100 for s3dis, scannet
# distance thresholds [m]
opt["distance_threshes"] = np.array([float("inf")])
# distance confidences
opt["distance_confs"] = np.array([-float("inf")])


def evaluate_matches(matches, match_criteria = match_criteria_M_pred):
    overlaps = opt["overlaps"]
    min_region_sizes = [opt["min_region_sizes"][0]]
    dist_threshes = [opt["distance_threshes"][0]]
    dist_confs = [opt["distance_confs"][0]]

    # results: class x overlap
    ap = np.zeros(
        (len(dist_threshes), len(CLASS_LABELS), len(overlaps)), float
    )
    for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
        zip(min_region_sizes, dist_threshes, dist_confs)
    ):
        for oi, overlap_th in enumerate(overlaps):
            pred_visited = {}
            for m in matches:
                for p in matches[m]["pred"]:
                    for label_name in CLASS_LABELS:
                        for p in matches[m]["pred"][label_name]:
                            if "uuid" in p:
                                pred_visited[p["uuid"]] = False
            for li, label_name in enumerate(CLASS_LABELS):
                y_true = np.empty(0)
                y_score = np.empty(0)
                hard_false_negatives = 0
                has_gt = False
                has_pred = False
                for m in matches:
                    pred_instances = matches[m]["pred"][label_name]
                    gt_instances = matches[m]["gt"][label_name]
                    # filter groups in ground truth
                    gt_instances = [
                        gt
                        for gt in gt_instances
                        if gt["instance_id"] >= 1000
                        and gt["vert_count"] >= min_region_size
                        and gt["med_dist"] <= distance_thresh
                        and gt["dist_conf"] >= distance_conf
                    ]
                    if gt_instances:
                        has_gt = True
                    if pred_instances:
                        has_pred = True

                    cur_true = np.ones(len(gt_instances))
                    cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                    cur_match = np.zeros(len(gt_instances), dtype=bool)
                    # collect matches
                    for (gti, gt) in enumerate(gt_instances):
                        found_match = False
                        num_pred = len(gt["matched_pred"])
                        for pred in gt["matched_pred"]:
                            # greedy assignments
                            if pred_visited[pred["uuid"]]:
                                continue
                            # overlap = float(pred["intersection"]) / (
                            #     gt["vert_count"]
                            #     + pred["vert_count"]
                            #     - pred["intersection"]
                            # )
                            # if overlap > overlap_th:
                            if match_criteria(pred, gt, overlap_th, False):
                                confidence = pred["confidence"]
                                # if already have a prediction for this gt,
                                # the prediction with the lower score is automatically a false positive
                                if cur_match[gti]:
                                    max_score = max(cur_score[gti], confidence)
                                    min_score = min(cur_score[gti], confidence)
                                    cur_score[gti] = max_score
                                    # append false positive
                                    cur_true = np.append(cur_true, 0)
                                    cur_score = np.append(cur_score, min_score)
                                    cur_match = np.append(cur_match, True)
                                # otherwise set score
                                else:
                                    found_match = True
                                    cur_match[gti] = True
                                    cur_score[gti] = confidence
                                    pred_visited[pred["uuid"]] = True
                        if not found_match:
                            hard_false_negatives += 1
                    # remove non-matched ground truth instances
                    cur_true = cur_true[cur_match == True]
                    cur_score = cur_score[cur_match == True]

                    # collect non-matched predictions as false positive
                    for pred in pred_instances:
                        found_gt = False
                        for gt in pred["matched_gt"]:
                            # overlap = float(gt["intersection"]) / (
                            #     gt["vert_count"]
                            #     + pred["vert_count"]
                            #     - gt["intersection"]
                            # )
                            # if overlap > overlap_th:
                            if match_criteria(gt, pred, overlap_th, True):
                                found_gt = True
                                break
                        if not found_gt:
                            num_ignore = pred["void_intersection"]
                            for gt in pred["matched_gt"]:
                                # group?
                                if gt["instance_id"] < 1000:
                                    num_ignore += gt["intersection"]
                                # small ground truth instances
                                if (
                                    gt["vert_count"] < min_region_size
                                    or gt["med_dist"] > distance_thresh
                                    or gt["dist_conf"] < distance_conf
                                ):
                                    num_ignore += gt["intersection"]
                            proportion_ignore = (
                                float(num_ignore) / pred["vert_count"]
                            )
                            # if not ignored append false positive
                            if proportion_ignore <= overlap_th:
                                cur_true = np.append(cur_true, 0)
                                confidence = pred["confidence"]
                                cur_score = np.append(cur_score, confidence)

                    # append to overall results
                    y_true = np.append(y_true, cur_true)
                    y_score = np.append(y_score, cur_score)

                # compute average precision
                if has_gt and has_pred:
                    # compute precision recall curve first

                    # sorting and cumsum
                    score_arg_sort = np.argsort(y_score)
                    y_score_sorted = y_score[score_arg_sort]
                    y_true_sorted = y_true[score_arg_sort]
                    y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                    # unique thresholds
                    (thresholds, unique_indices) = np.unique(
                        y_score_sorted, return_index=True
                    )
                    num_prec_recall = len(unique_indices) + 1

                    # prepare precision recall
                    num_examples = len(y_score_sorted)
                    # https://github.com/ScanNet/ScanNet/pull/26
                    # all predictions are non-matched but also all of them are ignored and not counted as FP
                    # y_true_sorted_cumsum is empty
                    # num_true_examples = y_true_sorted_cumsum[-1]
                    num_true_examples = (
                        y_true_sorted_cumsum[-1]
                        if len(y_true_sorted_cumsum) > 0
                        else 0
                    )
                    precision = np.zeros(num_prec_recall)
                    recall = np.zeros(num_prec_recall)

                    # deal with the first point
                    y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                    # deal with remaining
                    for idx_res, idx_scores in enumerate(unique_indices):
                        cumsum = y_true_sorted_cumsum[idx_scores - 1]
                        tp = num_true_examples - cumsum
                        fp = num_examples - idx_scores - tp
                        fn = cumsum + hard_false_negatives
                        p = float(tp) / (tp + fp)
                        r = float(tp) / (tp + fn)
                        precision[idx_res] = p
                        recall[idx_res] = r

                    # first point in curve is artificial
                    precision[-1] = 1.0
                    recall[-1] = 0.0

                    # compute average of precision-recall curve
                    recall_for_conv = np.copy(recall)
                    recall_for_conv = np.append(
                        recall_for_conv[0], recall_for_conv
                    )
                    recall_for_conv = np.append(recall_for_conv, 0.0)

                    stepWidths = np.convolve(
                        recall_for_conv, [-0.5, 0, 0.5], "valid"
                    )
                    # integrate is now simply a dot product
                    ap_current = np.dot(precision, stepWidths)

                elif has_gt:
                    ap_current = 0.0
                else:
                    ap_current = float("nan")
                ap[di, li, oi] = ap_current
    return ap


def compute_averages(aps):
    d_inf = 0
    o50 = np.where(np.isclose(opt["overlaps"], 0.5))
    o25 = np.where(np.isclose(opt["overlaps"], 0.25))
    oAllBut25 = np.where(np.logical_not(np.isclose(opt["overlaps"], 0.25)))
    avg_dict = {}
    # avg_dict['all_ap']     = np.nanmean(aps[ d_inf,:,:  ])
    avg_dict["all_ap"] = np.nanmean(aps[d_inf, :, oAllBut25])
    avg_dict["all_ap_50%"] = np.nanmean(aps[d_inf, :, o50])
    avg_dict["all_ap_25%"] = np.nanmean(aps[d_inf, :, o25])
    avg_dict["classes"] = {}
    for (li, label_name) in enumerate(CLASS_LABELS):
        avg_dict["classes"][label_name] = {}
        # avg_dict["classes"][label_name]["ap"]       = np.average(aps[ d_inf,li,  :])
        avg_dict["classes"][label_name]["ap"] = np.average(
            aps[d_inf, li, oAllBut25]
        )
        avg_dict["classes"][label_name]["ap50%"] = np.average(
            aps[d_inf, li, o50]
        )
        avg_dict["classes"][label_name]["ap25%"] = np.average(
            aps[d_inf, li, o25]
        )
    return avg_dict


def make_pred_info(pred: dict, 
                   eval_articulation: bool = False,
                   eval_hierarchy_inter: bool = False,):
    # pred = {'pred_scores' = 100, 'pred_classes' = 100 'pred_masks' = Nx100}
    # print("eval_hierarchy_inter in make_pred_info: ", eval_hierarchy_inter)
    pred_info = {}
    assert (
        pred["pred_classes"].shape[0]
        == pred["pred_scores"].shape[0]
        == pred["pred_masks"].shape[1]
    )
    if eval_hierarchy_inter:
        pred_interaction_mask = pred["pred_interaction_mask"]
        pred_interaction_vector = pred["pred_interaction_mask_vector"]
        pred_interaction_outs = pred["pred_interaction_out"]
        pred_mov_masks = pred["pred_masks"] > 1e-3
        pred_mov_inter_masks = pred_mov_masks & pred_interaction_vector

    for i in range(len(pred["pred_classes"])):
        pred_id = unique_ids.get_id()
        info = {}
        info["label_id"] = pred["pred_classes"][i]
        info["conf"] = pred["pred_scores"][i]
        info["mask"] = pred["pred_masks"][:, i] > 1e-3
        if eval_articulation:
            assert pred["pred_classes"].shape[0] == pred["pred_axises"].shape[0]
            info['origin'] = pred["pred_origins"][i]
            info['axis'] = pred["pred_axises"][i]
        if eval_hierarchy_inter:
            info['interaction_mask'] = pred_interaction_mask[:, i]
            info['interaction_vector_mask'] = pred_mov_inter_masks[:, i]
            info['interaction_out'] = pred_interaction_outs[:, i]
        pred_info[pred_id] = info  # we later need to identify these objects
        
        
    return pred_info


def assign_instances_for_scan(pred: dict, gt_file: str, 
        eval_articulation: bool = False, 
        gt_articulation: dict = None,
        eval_hierarchy_inter: bool = False):
    # print("eval_hierarchy_inter in assign_instances_for_scan: ", eval_hierarchy_inter)
    # print("gt_file: ", gt_file)
    # print(" Step 1: Load pred instances")
    pred_info = make_pred_info(pred, eval_articulation, eval_hierarchy_inter)
    # print("    - pred_info: ", len(pred_info))
    try:
        gt_ids = util_3d.load_ids(gt_file)
    except Exception as e:
        util.print_error("unable to load " + gt_file + ": " + str(e))

    # get gt instances
    all_gt_axis = None
    all_gt_origin = None
    if eval_articulation:
        all_gt_axis = {}
        all_gt_origin = {}
        for m in gt_articulation['articulations_dict']:
            all_gt_axis[m.item()] = gt_articulation['articulations_dict'][m]["axis"]
            all_gt_origin[m.item()] = gt_articulation['articulations_dict'][m]["origin"]
    interaction_labels = None
    if eval_hierarchy_inter:
        interaction_labels = gt_articulation['interaction_labels'] 
        
    gt_instances = util_3d.get_instances(
        gt_ids, VALID_CLASS_IDS, CLASS_LABELS, ID_TO_LABEL,
        all_gt_axis, all_gt_origin, interaction_labels
    )
    # print(" Step 2: Load gt instances")
    # print("    - gt_instances: ", len(gt_instances))
    # associate
    gt2pred = deepcopy(gt_instances)
    for label in gt2pred:
        for gt in gt2pred[label]:
            gt["matched_pred"] = []
    pred2gt = {}
    for label in CLASS_LABELS:
        pred2gt[label] = []
    num_pred_instances = 0
    # mask of void labels in the groundtruth
    bool_void = np.logical_not(np.in1d(gt_ids // 1000, VALID_CLASS_IDS))
    # go thru all prediction masks
    for uuid in pred_info:
        label_id = int(pred_info[uuid]["label_id"])
        conf = pred_info[uuid]["conf"]
        if not label_id in ID_TO_LABEL:
            continue
        label_name = ID_TO_LABEL[label_id]
        # read the mask
        pred_mask = pred_info[uuid]["mask"]
        assert len(pred_mask) == len(gt_ids)
        # convert to binary
        pred_mask = np.not_equal(pred_mask, 0)
        num = np.count_nonzero(pred_mask)
        if num < opt["min_region_sizes"][0]:
            continue  # skip if empty

        pred_instance = {}
        pred_instance["uuid"] = uuid
        pred_instance["pred_id"] = num_pred_instances
        pred_instance["label_id"] = label_id
        pred_instance["vert_count"] = num
        pred_instance["confidence"] = conf
        pred_instance["void_intersection"] = np.count_nonzero(
            np.logical_and(bool_void, pred_mask)
        )
        if eval_articulation:
            pred_instance["origin"] = pred_info[uuid]["origin"]
            pred_instance["axis"] = pred_info[uuid]["axis"]
        if eval_hierarchy_inter:
            pred_instance["interaction_mask"] = pred_info[uuid]["interaction_mask"]
            pred_instance["interaction_vector_mask"] = pred_info[uuid]["interaction_vector_mask"]
            pred_instance["interaction_out"] = pred_info[uuid]["interaction_out"]

        # matched gt instances
        matched_gt = []
        # go thru all gt instances with matching label
        for (gt_num, gt_inst) in enumerate(gt2pred[label_name]):
            intersection = np.count_nonzero(
                np.logical_and(gt_ids == gt_inst["instance_id"], pred_mask)
            )
            if intersection > 0:
                gt_copy = gt_inst.copy()
                pred_copy = pred_instance.copy()
                gt_copy["intersection"] = intersection
                pred_copy["intersection"] = intersection
                matched_gt.append(gt_copy)
                gt2pred[label_name][gt_num]["matched_pred"].append(pred_copy)
        pred_instance["matched_gt"] = matched_gt
        num_pred_instances += 1
        pred2gt[label_name].append(pred_instance)
    return gt2pred, pred2gt


def print_results(avgs, tag=''):
    sep = ""
    col1 = ":"
    lineLen = 64

    print("")
    print("#" * lineLen)
    line = ""
    line += '{:<15}'.format('what_{}'.format(tag)) + sep + col1
    line += "{:>15}".format("AP") + sep
    line += "{:>15}".format("AP_50%") + sep
    line += "{:>15}".format("AP_25%") + sep
    print(line)
    print("#" * lineLen)

    for (li, label_name) in enumerate(CLASS_LABELS):
        ap_avg = avgs["classes"][label_name]["ap"]
        ap_50o = avgs["classes"][label_name]["ap50%"]
        ap_25o = avgs["classes"][label_name]["ap25%"]
        line = "{:<15}".format(label_name) + sep + col1
        line += sep + "{:>15.3f}".format(ap_avg) + sep
        line += sep + "{:>15.3f}".format(ap_50o) + sep
        line += sep + "{:>15.3f}".format(ap_25o) + sep
        print(line)

    all_ap_avg = avgs["all_ap"]
    all_ap_50o = avgs["all_ap_50%"]
    all_ap_25o = avgs["all_ap_25%"]

    print("-" * lineLen)
    line = "{:<15}".format("average") + sep + col1
    line += "{:>15.3f}".format(all_ap_avg) + sep
    line += "{:>15.3f}".format(all_ap_50o) + sep
    line += "{:>15.3f}".format(all_ap_25o) + sep
    print(line)
    print("")


def write_result_file(avgs, filename):
    _SPLITTER = ","
    with open(filename, "w") as f:
        f.write(
            _SPLITTER.join(["class", "class id", "ap", "ap50", "ap25"]) + "\n"
        )
        for i in range(len(VALID_CLASS_IDS)):
            class_name = CLASS_LABELS[i]
            class_id = VALID_CLASS_IDS[i]
            ap = avgs["classes"][class_name]["ap"]
            ap50 = avgs["classes"][class_name]["ap50%"]
            ap25 = avgs["classes"][class_name]["ap25%"]
            f.write(
                _SPLITTER.join(
                    [str(x) for x in [class_name, class_id, ap, ap50, ap25]]
                )
                + "\n"
            )


def evaluate(
    preds: dict, gt_path: str, output_file: str, dataset: str = "scannet", 
    eval_articulation: bool = False, 
    gt_articulations: dict = None,
    eval_hierarchy_inter: bool = False,
):
    global CLASS_LABELS
    global VALID_CLASS_IDS
    global ID_TO_LABEL
    global LABEL_TO_ID
    global opt

    if dataset == "stpls3d":
        # global CLASS_LABELS
        # global VALID_CLASS_IDS
        # global ID_TO_LABEL
        # global LABEL_TO_ID

        opt["min_region_sizes"] = np.array([10])

        CLASS_LABELS = [
            "Build",
            "LowVeg",
            "MediumVeg",
            "HighVeg",
            "Vehicle",
            "Truck",
            "Aircraft",
            "MilitaryVeh",
            "Bike",
            "Motorcycle",
            "LightPole",
            "StreetSign",
            "Clutter",
            "Fence",
        ]
        VALID_CLASS_IDS = np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        )

        ID_TO_LABEL = {}
        LABEL_TO_ID = {}
        for i in range(len(VALID_CLASS_IDS)):
            LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
            ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]

    if dataset == "s3dis":
        # global CLASS_LABELS
        # global VALID_CLASS_IDS
        # global ID_TO_LABEL
        # global LABEL_TO_ID

        CLASS_LABELS = [
            "ceiling",
            "floor",
            "wall",
            "beam",
            "column",
            "window",
            "door",
            "table",
            "chair",
            "sofa",
            "bookcase",
            "board",
            "clutter",
        ]
        VALID_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
        ID_TO_LABEL = {}
        LABEL_TO_ID = {}
        for i in range(len(VALID_CLASS_IDS)):
            LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
            ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]
            
    if dataset == "scannetpp":
        CLASS_LABELS = ('table', 'door',
        'ceiling lamp', 'cabinet', 'blinds', 'curtain', 'chair', 'storage cabinet', 'office chair', 'bookshelf', 
        'whiteboard', 'window', 'box', 'monitor', 'shelf', 'heater', 'kitchen cabinet', 'sofa', 'bed', 'trash can', 'book',
        'plant', 'blanket', 'tv', 'computer tower', 'refrigerator', 'jacket', 'sink', 'bag', 'picture', 'pillow', 'towel',
        'suitcase', 'backpack', 'crate', 'keyboard', 'rack', 'toilet', 'printer', 'poster', 'painting', 'microwave', 'shoes',
        'socket', 'bottle', 'bucket', 'cushion', 'basket', 'shoe rack', 'telephone', 'file folder', 'laptop', 'plant pot',
        'exhaust fan', 'cup', 'coat hanger', 'light switch', 'speaker', 'table lamp', 'kettle', 'smoke detector', 'container',
        'power strip', 'slippers', 'paper bag', 'mouse', 'cutting board', 'toilet paper', 'paper towel', 'pot', 'clock',
        'pan', 'tap', 'jar', 'soap dispenser', 'binder', 'bowl', 'tissue box', 'whiteboard eraser', 'toilet brush', 
        'spray bottle', 'headphones', 'stapler', 'marker'
        )
        VALID_CLASS_IDS= np.array([17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 
        59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 
        85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100])
        ID_TO_LABEL = {}
        LABEL_TO_ID = {}
        for i in range(len(VALID_CLASS_IDS)):
            LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
            ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]
            
    if dataset == "multiscan" or dataset == "scenefun3d" or dataset == "articulate3d":
        CLASS_LABELS = ('rotation', 'translation')
        VALID_CLASS_IDS = np.array([1, 2])
        ID_TO_LABEL = {}
        LABEL_TO_ID = {}
        for i in range(len(VALID_CLASS_IDS)):
            LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
            ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]
            
    if dataset == "scannet200":
        CLASS_LABELS = (
            "chair",
            "table",
            "door",
            "couch",
            "cabinet",
            "shelf",
            "desk",
            "office chair",
            "bed",
            "pillow",
            "sink",
            "picture",
            "window",
            "toilet",
            "bookshelf",
            "monitor",
            "curtain",
            "book",
            "armchair",
            "coffee table",
            "box",
            "refrigerator",
            "lamp",
            "kitchen cabinet",
            "towel",
            "clothes",
            "tv",
            "nightstand",
            "counter",
            "dresser",
            "stool",
            "cushion",
            "plant",
            "ceiling",
            "bathtub",
            "end table",
            "dining table",
            "keyboard",
            "bag",
            "backpack",
            "toilet paper",
            "printer",
            "tv stand",
            "whiteboard",
            "blanket",
            "shower curtain",
            "trash can",
            "closet",
            "stairs",
            "microwave",
            "stove",
            "shoe",
            "computer tower",
            "bottle",
            "bin",
            "ottoman",
            "bench",
            "board",
            "washing machine",
            "mirror",
            "copier",
            "basket",
            "sofa chair",
            "file cabinet",
            "fan",
            "laptop",
            "shower",
            "paper",
            "person",
            "paper towel dispenser",
            "oven",
            "blinds",
            "rack",
            "plate",
            "blackboard",
            "piano",
            "suitcase",
            "rail",
            "radiator",
            "recycling bin",
            "container",
            "wardrobe",
            "soap dispenser",
            "telephone",
            "bucket",
            "clock",
            "stand",
            "light",
            "laundry basket",
            "pipe",
            "clothes dryer",
            "guitar",
            "toilet paper holder",
            "seat",
            "speaker",
            "column",
            "bicycle",
            "ladder",
            "bathroom stall",
            "shower wall",
            "cup",
            "jacket",
            "storage bin",
            "coffee maker",
            "dishwasher",
            "paper towel roll",
            "machine",
            "mat",
            "windowsill",
            "bar",
            "toaster",
            "bulletin board",
            "ironing board",
            "fireplace",
            "soap dish",
            "kitchen counter",
            "doorframe",
            "toilet paper dispenser",
            "mini fridge",
            "fire extinguisher",
            "ball",
            "hat",
            "shower curtain rod",
            "water cooler",
            "paper cutter",
            "tray",
            "shower door",
            "pillar",
            "ledge",
            "toaster oven",
            "mouse",
            "toilet seat cover dispenser",
            "furniture",
            "cart",
            "storage container",
            "scale",
            "tissue box",
            "light switch",
            "crate",
            "power outlet",
            "decoration",
            "sign",
            "projector",
            "closet door",
            "vacuum cleaner",
            "candle",
            "plunger",
            "stuffed animal",
            "headphones",
            "dish rack",
            "broom",
            "guitar case",
            "range hood",
            "dustpan",
            "hair dryer",
            "water bottle",
            "handicap bar",
            "purse",
            "vent",
            "shower floor",
            "water pitcher",
            "mailbox",
            "bowl",
            "paper bag",
            "alarm clock",
            "music stand",
            "projector screen",
            "divider",
            "laundry detergent",
            "bathroom counter",
            "object",
            "bathroom vanity",
            "closet wall",
            "laundry hamper",
            "bathroom stall door",
            "ceiling light",
            "trash bin",
            "dumbbell",
            "stair rail",
            "tube",
            "bathroom cabinet",
            "cd case",
            "closet rod",
            "coffee kettle",
            "structure",
            "shower head",
            "keyboard piano",
            "case of water bottles",
            "coat rack",
            "storage organizer",
            "folded chair",
            "fire alarm",
            "power strip",
            "calendar",
            "poster",
            "potted plant",
            "luggage",
            "mattress",
        )

        VALID_CLASS_IDS = np.array(
            (
                2,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                21,
                22,
                23,
                24,
                26,
                27,
                28,
                29,
                31,
                32,
                33,
                34,
                35,
                36,
                38,
                39,
                40,
                41,
                42,
                44,
                45,
                46,
                47,
                48,
                49,
                50,
                51,
                52,
                54,
                55,
                56,
                57,
                58,
                59,
                62,
                63,
                64,
                65,
                66,
                67,
                68,
                69,
                70,
                71,
                72,
                73,
                74,
                75,
                76,
                77,
                78,
                79,
                80,
                82,
                84,
                86,
                87,
                88,
                89,
                90,
                93,
                95,
                96,
                97,
                98,
                99,
                100,
                101,
                102,
                103,
                104,
                105,
                106,
                107,
                110,
                112,
                115,
                116,
                118,
                120,
                121,
                122,
                125,
                128,
                130,
                131,
                132,
                134,
                136,
                138,
                139,
                140,
                141,
                145,
                148,
                154,
                155,
                156,
                157,
                159,
                161,
                163,
                165,
                166,
                168,
                169,
                170,
                177,
                180,
                185,
                188,
                191,
                193,
                195,
                202,
                208,
                213,
                214,
                221,
                229,
                230,
                232,
                233,
                242,
                250,
                261,
                264,
                276,
                283,
                286,
                300,
                304,
                312,
                323,
                325,
                331,
                342,
                356,
                370,
                392,
                395,
                399,
                408,
                417,
                488,
                540,
                562,
                570,
                572,
                581,
                609,
                748,
                776,
                1156,
                1163,
                1164,
                1165,
                1166,
                1167,
                1168,
                1169,
                1170,
                1171,
                1172,
                1173,
                1174,
                1175,
                1176,
                1178,
                1179,
                1180,
                1181,
                1182,
                1183,
                1184,
                1185,
                1186,
                1187,
                1188,
                1189,
                1190,
                1191,
            )
        )

        ID_TO_LABEL = {}
        LABEL_TO_ID = {}
        for i in range(len(VALID_CLASS_IDS)):
            LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
            ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]

    total_true = 0
    total_seen = 0
    NUM_CLASSES = len(VALID_CLASS_IDS)

    true_positive_classes = np.zeros(NUM_CLASSES)
    positive_classes = np.zeros(NUM_CLASSES)
    gt_classes = np.zeros(NUM_CLASSES)

    # precision & recall
    total_gt_ins = np.zeros(NUM_CLASSES)
    at = 0.5
    tpsins = [[] for _ in range(NUM_CLASSES)]
    fpsins = [[] for _ in range(NUM_CLASSES)]
    # mucov and mwcov
    all_mean_cov = [[] for _ in range(NUM_CLASSES)]
    all_mean_weighted_cov = [[] for _ in range(NUM_CLASSES)]

    print("evaluating", len(preds), "scans...")
    matches = {}
    for i, (k, v) in enumerate(preds.items()):
        gt_file = os.path.join(gt_path, k + ".txt")
        if not os.path.isfile(gt_file):
            util.print_error(
                "Scan {} does not match any gt file {}".format(k, gt_file), user_fault=True
            )

        if dataset == "s3dis":
            gt_ids = util_3d.load_ids(gt_file)
            gt_sem = (gt_ids // 1000) - 1
            gt_ins = gt_ids - (gt_ids // 1000) * 1000

            # pred_sem = v['pred_classes'] - 1
            pred_sem = np.zeros(v["pred_masks"].shape[0], dtype=np.int)
            # TODO CONTINUE HERE!!!!!!!!!!!!!
            pred_ins = np.zeros(v["pred_masks"].shape[0], dtype=np.int)

            for inst_id in reversed(range(v["pred_masks"].shape[1])):
                point_ids = np.argwhere(v["pred_masks"][:, inst_id] == 1.0)[
                    :, 0
                ]
                pred_ins[point_ids] = inst_id + 1
                pred_sem[point_ids] = v["pred_classes"][inst_id] - 1

            # semantic acc
            total_true += np.sum(pred_sem == gt_sem)
            total_seen += pred_sem.shape[0]

            # TODO PARALLELIZ THIS!!!!!!!
            # pn semantic mIoU
            """
            for j in range(gt_sem.shape[0]):
                gt_l = int(gt_sem[j])
                pred_l = int(pred_sem[j])
                gt_classes[gt_l] += 1
                positive_classes[pred_l] += 1
                true_positive_classes[gt_l] += int(gt_l == pred_l)
            """

            uniq, counts = np.unique(pred_sem, return_counts=True)
            positive_classes[uniq] += counts

            uniq, counts = np.unique(gt_sem, return_counts=True)
            gt_classes[uniq] += counts

            uniq, counts = np.unique(
                gt_sem[pred_sem == gt_sem], return_counts=True
            )
            true_positive_classes[uniq] += counts

            # instance
            un = np.unique(pred_ins)
            pts_in_pred = [[] for _ in range(NUM_CLASSES)]
            for ig, g in enumerate(un):  # each object in prediction
                if g == -1:
                    continue
                tmp = pred_ins == g
                sem_seg_i = int(stats.mode(pred_sem[tmp])[0])
                pts_in_pred[sem_seg_i] += [tmp]

            un = np.unique(gt_ins)
            pts_in_gt = [[] for _ in range(NUM_CLASSES)]
            for ig, g in enumerate(un):
                tmp = gt_ins == g
                sem_seg_i = int(stats.mode(gt_sem[tmp])[0])
                pts_in_gt[sem_seg_i] += [tmp]

            # instance mucov & mwcov
            for i_sem in range(NUM_CLASSES):
                sum_cov = 0
                mean_cov = 0
                mean_weighted_cov = 0
                num_gt_point = 0
                for ig, ins_gt in enumerate(pts_in_gt[i_sem]):
                    ovmax = 0.0
                    num_ins_gt_point = np.sum(ins_gt)
                    num_gt_point += num_ins_gt_point
                    for ip, ins_pred in enumerate(pts_in_pred[i_sem]):
                        union = ins_pred | ins_gt
                        intersect = ins_pred & ins_gt
                        iou = float(np.sum(intersect)) / np.sum(union)

                        if iou > ovmax:
                            ovmax = iou
                            ipmax = ip

                    sum_cov += ovmax
                    mean_weighted_cov += ovmax * num_ins_gt_point

                if len(pts_in_gt[i_sem]) != 0:
                    mean_cov = sum_cov / len(pts_in_gt[i_sem])
                    all_mean_cov[i_sem].append(mean_cov)

                    mean_weighted_cov /= num_gt_point
                    all_mean_weighted_cov[i_sem].append(mean_weighted_cov)

        if dataset == "s3dis":
            # instance precision & recall
            for i_sem in range(NUM_CLASSES):
                tp = [0.0] * len(pts_in_pred[i_sem])
                fp = [0.0] * len(pts_in_pred[i_sem])
                gtflag = np.zeros(len(pts_in_gt[i_sem]))
                total_gt_ins[i_sem] += len(pts_in_gt[i_sem])

                for ip, ins_pred in enumerate(pts_in_pred[i_sem]):
                    ovmax = -1.0

                    for ig, ins_gt in enumerate(pts_in_gt[i_sem]):
                        union = ins_pred | ins_gt
                        intersect = ins_pred & ins_gt
                        iou = float(np.sum(intersect)) / np.sum(union)

                        if iou > ovmax:
                            ovmax = iou
                            igmax = ig

                    if ovmax >= at:
                        tp[ip] = 1  # true
                    else:
                        fp[ip] = 1  # false positive

                tpsins[i_sem] += tp
                fpsins[i_sem] += fp

        matches_key = os.path.abspath(gt_file)
        # assign gt to predictions
        if (dataset == "multiscan" or dataset == "scenefun3d" or dataset == 'articulate3d') and (eval_articulation or eval_hierarchy_inter):
            # print("eval_hierarchy_inter in evaluate: ", eval_hierarchy_inter)
            gt2pred, pred2gt = assign_instances_for_scan(v, gt_file, 
                                eval_articulation, 
                                gt_articulations[k],
                                eval_hierarchy_inter)
        else:
            gt2pred, pred2gt = assign_instances_for_scan(v, gt_file)
        matches[matches_key] = {}
        matches[matches_key]["gt"] = gt2pred
        matches[matches_key]["pred"] = pred2gt
        sys.stdout.write("\rscans processed: {}".format(i + 1))
        sys.stdout.flush()
    print("")
    ap_scores = evaluate_matches(matches)
    avgs = compute_averages(ap_scores)
    # print
    print_results(avgs)
    write_result_file(avgs, output_file)
    
    if (dataset == "multiscan" or \
        dataset == "scenefun3d" and \
        eval_articulation):
        ap_scores_MA = evaluate_matches(matches, match_criteria_MA_pred)
        ap_scores_MO = evaluate_matches(matches, match_criteria_MO_pred)
        ap_scores_MAO = evaluate_matches(matches, match_criteria_MAO_pred)
        ap_scores_MAO_ST = evaluate_matches(matches, match_criteria_MAO_pred_standard)
        
        avgs_MA = compute_averages(ap_scores_MA)
        avgs_MAO = compute_averages(ap_scores_MAO)
        avgs_MAO_ST = compute_averages(ap_scores_MAO_ST)
        avgs_MO = compute_averages(ap_scores_MO)
        
        print_results(avgs_MA, "MA")
        print_results(avgs_MO, "MO")
        print_results(avgs_MAO, "MAO")
        print_results(avgs_MAO_ST, "MAO_ST")
        
        return avgs['all_ap_50%'], avgs_MA['all_ap_50%'], avgs_MO['all_ap_50%'], avgs_MAO['all_ap_50%'],             avgs_MAO_ST['all_ap_50%']
    
    if dataset == 'articulate3d':
        if eval_articulation or eval_hierarchy_inter:
            if eval_articulation:
                ap_scores_MA = evaluate_matches(matches, match_criteria_MA_pred)
                ap_scores_MO = evaluate_matches(matches, match_criteria_MO_pred)
                ap_scores_MAO = evaluate_matches(matches, match_criteria_MAO_pred)
                ap_scores_MAO_ST = evaluate_matches(matches, match_criteria_MAO_pred_standard)
                
                avgs_MA = compute_averages(ap_scores_MA)
                avgs_MAO = compute_averages(ap_scores_MAO)
                avgs_MAO_ST = compute_averages(ap_scores_MAO_ST)
                avgs_MO = compute_averages(ap_scores_MO)
                
                print_results(avgs_MA, "MA")
                print_results(avgs_MO, "MO")
                print_results(avgs_MAO, "MAO")
                print_results(avgs_MAO_ST, "MAO_ST")
            if eval_hierarchy_inter:
                ap_scores_I = evaluate_matches(matches, match_criteria_I)
                avs_I = compute_averages(ap_scores_I)
                print_results(avs_I, "I")
                
                ap_scores_I_vector = evaluate_matches(matches, match_criteria_I_vector)
                avgs_I_vector = compute_averages(ap_scores_I_vector)
                print_results(avgs_I_vector, "I_vector")
                
                ap_scores_I_out = evaluate_matches(matches, match_criteria_I_out)
                avgs_I_out = compute_averages(ap_scores_I_out)
                print_results(avgs_I_out, "I_out")
                
                ap_scores_I_GT = evaluate_matches(matches, match_criteria_I_GT)
                avs_I_GT = compute_averages(ap_scores_I_GT)
                print_results(avs_I_GT, "I_GT")
                
                ap_scores_I_out_GT = evaluate_matches(matches, match_criteria_I_out_GT)
                avs_I_out_GT = compute_averages(ap_scores_I_out_GT)
                print_results(avs_I_out_GT, "I_out_GT")
                
                
            MA_ap50 = avgs_MA['all_ap_50%'] if eval_articulation else 0.
            MO_ap50 = avgs_MO['all_ap_50%'] if eval_articulation else 0.
            MAO_ap50 = avgs_MAO['all_ap_50%'] if eval_articulation else 0.
            MAO_ST_ap50 = avgs_MAO_ST['all_ap_50%'] if eval_articulation else 0.
            I_ap50 = avs_I['all_ap_50%'] if eval_hierarchy_inter else 0.
            I_vector_ap50 = avgs_I_vector['all_ap_50%'] if eval_hierarchy_inter else 0.
            I_out_ap50 = avgs_I_out['all_ap_50%'] if eval_hierarchy_inter else 0.
            I_GT_ap50 = avs_I_GT['all_ap_50%'] if eval_hierarchy_inter else 0.
            I_out_GT_ap50 = avs_I_out_GT['all_ap_50%'] if eval_hierarchy_inter else 0.
            return avgs['all_ap_50%'], MA_ap50, MO_ap50, MAO_ap50, MAO_ST_ap50, I_ap50, I_vector_ap50, I_out_ap50, I_GT_ap50, I_out_GT_ap50
        else:
            return avgs['all_ap_50%']
        # return avgs['all_ap_50%'], avgs_MA['all_ap_50%'], avgs_MO['all_ap_50%'], avgs_MAO['all_ap_50%'], avgs_MAO_ST['all_ap_50%'], 0., 0.

    if dataset == "s3dis":
        MUCov = np.zeros(NUM_CLASSES)
        MWCov = np.zeros(NUM_CLASSES)
        for i_sem in range(NUM_CLASSES):
            MUCov[i_sem] = np.mean(all_mean_cov[i_sem])
            MWCov[i_sem] = np.mean(all_mean_weighted_cov[i_sem])

        precision = np.zeros(NUM_CLASSES)
        recall = np.zeros(NUM_CLASSES)
        for i_sem in range(NUM_CLASSES):
            tp = np.asarray(tpsins[i_sem]).astype(np.float)
            fp = np.asarray(fpsins[i_sem]).astype(np.float)
            tp = np.sum(tp)
            fp = np.sum(fp)
            rec = tp / total_gt_ins[i_sem]
            prec = tp / (tp + fp)

            precision[i_sem] = prec
            recall[i_sem] = rec

        """
        LOG_FOUT = open(os.path.join('results_a5.txt'), 'w')
    
        def log_string(out_str):
            LOG_FOUT.write(out_str + '\n')
            LOG_FOUT.flush()
            print(out_str)
        """

        return np.mean(precision), np.mean(recall)


# TODO: remove this
# import pandas as pd
# def main():
#    print("!!! CLI is only for debugging purposes. use `evaluate()` instead.")
#    evaluate(pd.read_pickle("/globalwork/schult/saved_predictions.pkl"), opt.gt_path, opt.output_file)

# if __name__ == '__main__':
#    main()
