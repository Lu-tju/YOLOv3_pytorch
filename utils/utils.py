import glob
import random
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import torch_utils

# Set printoptions
torch.set_printoptions(linewidth=1320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5


def float3(x):  # format floats to 3 decimals
    return float(format(x, '.3f'))


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch_utils.init_seeds(seed=seed)


def load_classes(path):
    # Loads class labels at 'path'
    fp = open(path, 'r')
    names = fp.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


def model_info(model):
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %38s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %38s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients' % (i + 1, n_p, n_g))


def coco_class_weights():  # frequency of each class in coco train2014
    weights = 1 / torch.FloatTensor(
        [187437, 4955, 30920, 6033, 3838, 4332, 3160, 7051, 7677, 9167, 1316, 1372, 833, 6757, 7355, 3302, 3776, 4671,
         6769, 5706, 3908, 903, 3686, 3596, 6200, 7920, 8779, 4505, 4272, 1862, 4698, 1962, 4403, 6659, 2402, 2689,
         4012, 4175, 3411, 17048, 5637, 14553, 3923, 5539, 4289, 10084, 7018, 4314, 3099, 4638, 4939, 5543, 2038, 4004,
         5053, 4578, 27292, 4113, 5931, 2905, 11174, 2873, 4036, 3415, 1517, 4122, 1980, 4464, 1190, 2302, 156, 3933,
         1877, 17630, 4337, 4624, 1075, 3468, 135, 1380])
    weights /= weights.sum()
    return weights


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    # 画bbox框
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.03)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.03)
        torch.nn.init.constant_(m.bias.data, 0.0)


def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros_like(x) if x.dtype is torch.float32 else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros_like(x) if x.dtype is torch.float32 else np.zeros_like(x)
    y[:, 0] = (x[:, 0] - x[:, 2] / 2)
    y[:, 1] = (x[:, 1] - x[:, 3] / 2)
    y[:, 2] = (x[:, 0] + x[:, 2] / 2)
    y[:, 3] = (x[:, 1] + x[:, 3] / 2)
    return y


def scale_coords(img_size, coords, img0_shape):
    # Rescale x1, y1, x2, y2 from 416 to image size
    gain = float(img_size) / max(img0_shape)  # gain  = old / new
    pad_x = (img_size - img0_shape[1] * gain) / 2  # width padding
    pad_y = (img_size - img0_shape[0] * gain) / 2  # height padding
    coords[:, [0, 2]] -= pad_x
    coords[:, [1, 3]] -= pad_y
    coords[:, :4] /= gain
    coords[:, :4] = torch.clamp(coords[:, :4], min=0)
    return coords


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ 计算平均精度，给出召回率和精度曲线。
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).分类为的正样本，且分对了的 个数
        conf:  Objectness value from 0-1 (list).边框置信度
        pred_cls: Predicted object classes (list).预测类
        target_cls: True object classes (list).实际类
    # Returns
        平均精度。
    """

    # 按置信度降序排列
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes（np.unique：除去重复元素）
    unique_classes = np.unique(np.concatenate((pred_cls, target_cls), 0))

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = sum(target_cls == c)  # 实际对象数
        n_p = sum(i)  # 预测对象数

        if (n_p == 0) and (n_gt == 0):
            continue
        elif (n_p == 0) or (n_gt == 0):
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs    假设这一类都是识别为人
            # False Positives  窗户认作人
            # True Positives   人认作了人
            fpc = np.cumsum(1 - tp[i])
            tpc = np.cumsum(tp[i])

            # Recall：分类器识别为人，且确实是人，占所有‘人’ 的比例
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(tpc[-1] / (n_gt + 1e-16))

            # Precision：分类器认为是人，且确实是人，占分类器认为是人的比例
            precision_curve = tpc / (tpc + fpc)
            p.append(tpc[-1] / (tpc[-1] + fpc[-1]))

            # 从R-P曲线得到曲线下面积AP
            ap.append(compute_ap(recall_curve, precision_curve))

    return np.array(ap), unique_classes.astype('int32'), np.array(r), np.array(p)


def compute_ap(recall, precision):
    """ 计算AP，给出召回率和精度曲线。
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
       平均精度
    """
    # correct AP calculation
    # 首先添加起止点，first append sentinel values at the end
    # 点对：[recall, precision]，如最简单的，图片一个目标，就是[0,1] [recall,precision] [1,0],对应：rec=[0,recall,1],pre=[1,precision,0]

    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))  # 最后处理成了[0,pre,1]为何如此原因未知

    # 计算精度包络线
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # 要计算PR曲线下的面积, look for points
    # rec坐标（横轴）哪里变化了
    i = np.where(mrec[1:] != mrec[:-1])[0]  # 就是一个从第0元素到最后，一个从第1元素到最后，如果哪个位置不等，则变化了

    # 求所有小块矩形面积(Delta rec) * pre并求和，得到R-P曲线近似积分
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def bbox_iou(box1, box2, x1y1x2y2=True):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:
        # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        # x, y, w, h = box1
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                 (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1) + 1e-16) + \
                 (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area

    return inter_area / union_area  # iou


def wh_iou(box1, box2):
    # 计算IOU
    # Returns the IoU of wh1 to wh2. wh1 is 2, wh2 is nx2
    box2 = box2.t()

    # w, h = box1
    w1, h1 = box1[0], box1[1]
    w2, h2 = box2[0], box2[1]

    # Intersection area
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)

    # Union Area
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area

    return inter_area / union_area  # iou


def compute_loss(p, targets):  # predictions, targets
    FT = torch.cuda.FloatTensor if p[0].is_cuda else torch.FloatTensor
    loss, lxy, lwh, lcls, lconf = FT([0]), FT([0]), FT([0]), FT([0]), FT([0])
    txy, twh, tcls, tconf, indices = targets
    MSE = nn.MSELoss()
    CE = nn.CrossEntropyLoss()
    BCE = nn.BCEWithLogitsLoss()

    # Compute losses
    # gp = [x.numel() for x in tconf]  # grid points
    for i, pi0 in enumerate(p):  # layer i predictions, i   P:两个元组：[4,3,13,13,85]、[4,3,26,26,85]
        b, a, gj, gi = indices[i]  # 目标第i个尺度的 image, anchor, gridx, gridy

        # Compute losses
        k = 1  # nT / bs
        # 负责预测该物体的anchor的误差：
        if len(b) > 0:
            pi = pi0[b, a, gj, gi]  # 找到perd中负责预测该物体的anchor，[目标数，85]，这个anchor和他的4个坐标+C+80类
            lxy += k * MSE(torch.sigmoid(pi[..., 0:2]), txy[i])  # xy:方均根误差
            lwh += k * MSE(pi[..., 2:4], twh[i])  # wh: 方均根误差
            lcls += (k / 4) * CE(pi[..., 5:], tcls[i])  # 80类分类结果: 交叉熵误差   p*log(q) p:期望1, q:实际置信度

        # pos_weight = FT([gp[i] / min(gp) * 4.])
        # BCE = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # 所有的预测都要计算的误差C：pi0（看看tconf格式）
        lconf += (k * 64) * BCE(pi0[..., 4], tconf[i])  # 二分类交叉熵，如置信度C=0.8，就计算[0.8,0.2]和[1,0]之间的交叉熵误差
    loss = lxy + lwh + lconf + lcls

    # Add to dictionary
    d = defaultdict(float)
    losses = [loss.item(), lxy.item(), lwh.item(), lconf.item(), lcls.item()]
    for name, x in zip(['total', 'xy', 'wh', 'conf', 'cls'], losses):
        d[name] = x

    return loss, d


def build_targets(model, targets, pred):

    """构建训练标签"""

    # targets = [image序号, 类, x, y, w, h]
    if isinstance(model, nn.DataParallel):
        model = model.module
    yolo_layers = get_yolo_layers(model)    # yolo层的序号：16，23

    # anchors = closest_anchor(model, targets)  # [layer, anchor, i, j]
    txy, twh, tcls, tconf, indices = [], [], [], [], []

    # 正负样本选取
    for i, layer in enumerate(yolo_layers):     # 好几个loyo层嘛，挨个循环
        nG = model.module_list[layer][0].nG  # grid size = 13
        anchor_vec = model.module_list[layer][0].anchor_vec     # anchor相对grid的大小

        # iou最大的anchor负责预测这个目标，填上位置类别信息，其余的位置归0
        # iou of targets-anchors
        gwh = targets[:, 4:6] * nG  # 目标相对grid的W H  targets是以原图归一化的，而预测以grid归一化，相差了nG=13倍
        iou = [wh_iou(x, gwh) for x in anchor_vec]  # 计算IOU
        iou, a = torch.stack(iou, 0).max(0)  # best iou and anchor: 由3个anchor中的哪个来预测（应该所有的anchor找吧）

        # 去掉小于阈值的IOU，我改的：为了防止尺度预测混乱
        reject = True
        if reject:
            j = iou > 0.5   # 默认0.1
            t, a, gwh = targets[j], a[j], gwh[j]    # t：targets信息，一个bbox5个元素；a:哪个anchor预测；gwh：目标相对grid的位置
        else:
            t = targets

        # Indices：anchor的索引：包含四个元组，分别是：哪张图、哪个anchor、中心在哪个grid（x,y）
        b, c = t[:, 0:2].long().t()  # b: target image 哪张图  c: class
        gxy = t[:, 2:4] * nG    # 目标的X Y *13，变成相对grid的X Y
        gi, gj = gxy.long().t()  # grid_i, grid_j， 它的整数部分，即，它落在哪个grid中。如[2.5, 3.6]说明他中心在（2，3）这个grid
        indices.append((b, a, gj, gi))

        # tx、ty
        # 为什么不反sigmoid?因为在loss计算中，pred进行了simoid，这里就不用反了
        txy.append(gxy - gxy.floor())   # 目标中心相对grid左上角位置,每个坐标一行[目标数n,2]

        # Width and height
        twh.append(torch.log(gwh / anchor_vec[a]))  # 公式得：tw、th
        # twh.append(torch.sqrt(gwh / anchor_vec[a]) / 2)  # power method

        # Class
        tcls.append(c)      # [目标数n,1]

        # Conf
        tci = torch.zeros_like(pred[i][..., 0])     # [batch, 3, 13, 13] : [哪张图，哪个anchor，x，y]
        tci[b, a, gj, gi] = 1  # conf：设置 负责预测这个物体的anchor 所在的grid的 conf=1
        tconf.append(tci)

    # 这样下来， txy, twh, tcls, tconf, indices各包含2个元组，对应2种尺度（yolo tiny）下的预测情况（会有重复，一个尺度预测了，另一个尺度还预测它）

    return txy, twh, tcls, tconf, indices


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    非极大值抑制 NMS
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    output = [None for _ in range(len(prediction))]
    for image_i, pred in enumerate(prediction):
        # Experiment: Prior class size rejection
        # x, y, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        # a = w * h  # area
        # ar = w / (h + 1e-16)  # aspect ratio
        # n = len(w)
        # log_w, log_h, log_a, log_ar = torch.log(w), torch.log(h), torch.log(a), torch.log(ar)
        # shape_likelihood = np.zeros((n, 60), dtype=np.float32)
        # x = np.concatenate((log_w.reshape(-1, 1), log_h.reshape(-1, 1)), 1)
        # from scipy.stats import multivariate_normal
        # for c in range(60):
        # shape_likelihood[:, c] =
        #   multivariate_normal.pdf(x, mean=mat['class_mu'][c, :2], cov=mat['class_cov'][c, :2, :2])

        # Filter out confidence scores below threshold
        class_prob, class_pred = torch.max(F.softmax(pred[:, 5:], 1), 1)
        v = pred[:, 4] > conf_thres
        v = v.nonzero().squeeze()
        if len(v.shape) == 0:
            v = v.unsqueeze(0)

        pred = pred[v]
        class_prob = class_prob[v]
        class_pred = class_pred[v]

        # If none are remaining => process next image
        nP = pred.shape[0]
        if not nP:
            continue

        # From (center x, center y, width, height) to (x1, y1, x2, y2)
        pred[:, :4] = xywh2xyxy(pred[:, :4])

        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_prob, class_pred)
        detections = torch.cat((pred[:, :5], class_prob.float().unsqueeze(1), class_pred.float().unsqueeze(1)), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique().to(prediction.device)

        nms_style = 'OR'  # 'OR' (default), 'AND', 'MERGE' (experimental)
        for c in unique_labels:
            # Get the detections with class c
            dc = detections[detections[:, -1] == c]
            # Sort the detections by maximum object confidence
            _, conf_sort_index = torch.sort(dc[:, 4] * dc[:, 5], descending=True)
            dc = dc[conf_sort_index]

            # Non-maximum suppression
            det_max = []
            ind = list(range(len(dc)))
            if nms_style == 'OR':  # default
                while len(ind):
                    j = ind[0]
                    det_max.append(dc[j:j + 1])  # save highest conf detection
                    reject = bbox_iou(dc[j], dc[ind]) > nms_thres
                    [ind.pop(i) for i in reversed(reject.nonzero())]
                # while dc.shape[0]:  # SLOWER METHOD
                #     det_max.append(dc[:1])  # save highest conf detection
                #     if len(dc) == 1:  # Stop if we're at the last detection
                #         break
                #     iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                #     dc = dc[1:][iou < nms_thres]  # remove ious > threshold

                # Image      Total          P          R        mAP
                #  4964       5000      0.629      0.594      0.586

            elif nms_style == 'AND':  # requires overlap, single boxes erased
                while len(dc) > 1:
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    if iou.max() > 0.5:
                        det_max.append(dc[:1])
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold

            elif nms_style == 'MERGE':  # weighted mixture box
                while len(dc) > 0:
                    iou = bbox_iou(dc[0], dc[0:])  # iou with other boxes
                    i = iou > nms_thres

                    weights = dc[i, 4:5] * dc[i, 5:6]
                    dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()
                    det_max.append(dc[:1])
                    dc = dc[iou < nms_thres]

                # Image      Total          P          R        mAP
                #  4964       5000      0.633      0.598      0.589  # normal

            if len(det_max) > 0:
                det_max = torch.cat(det_max)
                # Add max detections to outputs
                output[image_i] = det_max if output[image_i] is None else torch.cat((output[image_i], det_max))

    return output


def get_yolo_layers(model):
    bool_vec = [x['type'] == 'yolo' for x in model.module_defs]
    return [i for i, x in enumerate(bool_vec) if x]  # [82, 94, 106] for yolov3


def return_torch_unique_index(u, uv):
    n = uv.shape[1]  # number of columns
    first_unique = torch.zeros(n, device=u.device).long()
    for j in range(n):
        first_unique[j] = (uv[:, j:j + 1] == u).all(0).nonzero()[0]

    return first_unique


def strip_optimizer_from_checkpoint(filename='weights/best.pt'):
    # Strip optimizer from *.pt files for lighter files (reduced by 2/3 size)
    a = torch.load(filename, map_location='cpu')
    a['optimizer'] = []
    torch.save(a, filename.replace('.pt', '_lite.pt'))


def coco_class_count(path='../coco/labels/train2014/'):
    # Histogram of occurrences per class
    nC = 80  # number classes
    x = np.zeros(nC, dtype='int32')
    files = sorted(glob.glob('%s/*.*' % path))
    for i, file in enumerate(files):
        labels = np.loadtxt(file, dtype=np.float32).reshape(-1, 5)
        x += np.bincount(labels[:, 0].astype('int32'), minlength=nC)
        print(i, len(files))


def coco_only_people(path='../coco/labels/val2014/'):
    # Find images with only people
    files = sorted(glob.glob('%s/*.*' % path))
    for i, file in enumerate(files):
        labels = np.loadtxt(file, dtype=np.float32).reshape(-1, 5)
        if all(labels[:, 0] == 0):
            print(labels.shape[0], file)


def plot_results(start=0):
    # Plot YOLO training results file 'results.txt'
    # import os; os.system('wget https://storage.googleapis.com/ultralytics/yolov3/results_v3.txt')
    # from utils.utils import *; plot_results()

    plt.figure(figsize=(14, 7))
    s = ['X + Y', 'Width + Height', 'Confidence', 'Classification', 'Total Loss', 'Precision', 'Recall', 'mAP']
    files = sorted(glob.glob('results*.txt'))
    for f in files:
        results = np.loadtxt(f, usecols=[2, 3, 4, 5, 6, 9, 10, 11]).T  # column 11 is mAP
        x = range(1, results.shape[1])
        for i in range(8):
            plt.subplot(2, 4, i + 1)
            plt.plot(results[i, x[start:]], marker='.', label=f)
            plt.title(s[i])
            if i == 0:
                plt.legend()
