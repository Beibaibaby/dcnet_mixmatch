import logging
import os
from models.model_factory import build_model
import json
from utils.metrics import Accuracy
from utils.cam_utils import *
from sklearn.metrics import confusion_matrix
from utils.model_utils import load_checkpoint


def get_binary_ious_conf_matrix(gt_mask, pred_mask, all_thresholds=np.arange(0, 1, 1 / 10)):
    """
    Assuming binary gt_mask and a prediction mask between 0-1, it computes IOU at different thresholds

    :param gt_mask: binary numpy vector, size=HW
    :param pred_mask: binary numpy vector, size=HW (between 0 and 1)
    :return:
    """
    ious, conf_matrices, thresholds = [], [], []
    labels = [0, 1]

    for thresh in all_thresholds:
        bin_pred_mask = (pred_mask > thresh).astype(np.int)
        conf_matrix_fn = RunningConfusionMatrix(labels=labels)

        conf_matrix_fn.update_matrix(gt_mask, bin_pred_mask)

        iou = conf_matrix_fn.compute_current_mean_intersection_over_union()

        ious.append(iou)
        conf_matrices.append(conf_matrix_fn.overall_confusion_matrix)
        thresholds.append(thresh)
    return ious, conf_matrices, thresholds


class RunningConfusionMatrix():
    # Adapted from: https://github.com/isi-vista/structure_via_consensus/blob/master/src_release/conf_matrix.py
    def __init__(self, labels, ignore_label=0):
        """
        Updatable confusion matrix
        :param labels: List[int] representing class ids
        :param ignore_label: Class
        """
        self.labels = labels
        self.ignore_label = ignore_label
        self.overall_confusion_matrix = None

    def update_matrix(self, ground_truth, prediction):
        """
        Updates the confusion matrix
        :param ground_truth: Sequence of ground truth values, shape: [N]
        :param prediction: Sequence of predicted values, shape: [N]
        :return:
        """

        # Mask-out value is ignored by default in the sklearn
        # read sources to see how that was handled
        # But sometimes all the elements in the groundtruth can
        # be equal to ignore value which will cause the crush
        # of scikit_learn.confusion_matrix(), this is why we check it here
        if (ground_truth == self.ignore_label).all():
            return
        current_confusion_matrix = confusion_matrix(y_true=ground_truth,
                                                    y_pred=prediction,
                                                    labels=self.labels)

        if self.overall_confusion_matrix is not None:
            self.overall_confusion_matrix += current_confusion_matrix
        else:

            self.overall_confusion_matrix = current_confusion_matrix

    def compute_current_mean_intersection_over_union(self):
        """
        Computes mean IOU using intersection between ground truth and prediction
        :return:
        """
        intersection = np.diag(self.overall_confusion_matrix)
        ground_truth_set = self.overall_confusion_matrix.sum(axis=1)
        predicted_set = self.overall_confusion_matrix.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection

        intersection_over_union = intersection / union.astype(np.float32)
        mean_intersection_over_union = np.mean(intersection_over_union)

        return mean_intersection_over_union


def main_calc_segmentation_metrics(config, data_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(config.model)
    model = model.to(device)
    save_dir = '/'.join(config.checkpoint_path.split('/')[0:-1]) + '/sample_masks'
    if config.checkpoint_path is not None:
        load_checkpoint(model, config.checkpoint_path)
    calc_segmentation_metrics(model, data_loader, device, save_dir, config.dataset.num_classes)


# def calc_segmentation_metrics(model, data_loader, device, save_dir, num_classes, exit_to_resize_to=2,
#                               save_every=5):
#     # TODO: working on migrating to SemanticSegmentationMetrics class
#     """
#
#     :param model: Supports both OccamNets and non-Occam networks. For the latter, specify the exit layer in
#     cam_utils.get_target_layers()
#     :param data_loader: batch must contain 'x', 'y' and 'mask'
#     :param device:
#     :param save_dir:
#     :param num_classes:
#     :param exit_to_resize_to: CAMs will be resized to the CAMs from this exit
#     :return:
#     """
#     exit_to_thresh_to_details = {}
#     _viz_ix = 0
#     exit_to_acc_metric = {}
#     model = model.eval()
#
#     for batch_ix, batch in enumerate(data_loader):
#         # Gather GT class cams
#         exit_to_gt_cams, model_out = get_gt_class_cams(model, batch, device=device, target_exit_size=exit_to_resize_to)
#         exit_to_cams = {}
#
#         if 'mask' in batch:
#             gt_full_masks = batch['mask'].mean(dim=1).squeeze().detach().cpu().unsqueeze(1)
#             # GT segmentation mask
#             if 'occam' in type(model).__name__.lower():
#                 resize_H, resize_W = exit_to_gt_cams[exit_to_resize_to].shape[1], \
#                                      exit_to_gt_cams[exit_to_resize_to].shape[2]
#                 gt_masks = interpolate(gt_full_masks, resize_H, resize_W)
#             else:
#                 resize_H, resize_W = 14, 14
#                 gt_masks = interpolate(gt_full_masks, resize_H, resize_W)
#         else:
#             raise Exception("'mask' field not found: Data loader did not return the ground truth mask!")
#
#         # Gather logits for accuracy computation and CAMs for segmentation metrics
#         for exit_ix in exit_to_gt_cams:
#             key = f"E={exit_ix}, cam"
#             if isinstance(model_out, dict) and key in model_out:
#                 exit_to_cams[exit_ix] = model_out[f"E={exit_ix}, cam"]
#
#             # Initialize the accuracy metric per exit
#             if exit_ix not in exit_to_thresh_to_details:
#                 exit_to_thresh_to_details[exit_ix] = {}
#                 exit_to_acc_metric[exit_ix] = Accuracy()
#
#             # Gather class-wise logits
#             if isinstance(model_out, dict):
#                 if f"E={exit_ix}, logits" in model_out:
#                     logits = model_out[f"E={exit_ix}, logits"]
#                 elif exit_ix == 'early_exit':
#                     logits = model_out['E=early, logits']
#                 else:
#                     logits = model_out['logits']
#             else:
#                 logits = model_out
#             exit_to_gt_cams[exit_ix] = exit_to_gt_cams[exit_ix].unsqueeze(1).detach().cpu()
#
#             # Pass CAMs through a sigmoid layer
#             gt_class_pred_mask = torch.sigmoid(exit_to_gt_cams[exit_ix])
#             gt_class_pred_mask = interpolate(gt_class_pred_mask, resize_H, resize_W)
#
#             # Compute IOUs at different thresholds (instead of only using a threshold of 0.5)
#             ious, conf_matrices, thresholds, = get_binary_ious_conf_matrix(
#                 gt_masks.detach().cpu().long().flatten().numpy(),
#                 gt_class_pred_mask.detach().cpu().flatten().numpy())
#
#             for iou, conf_matrix, thres in zip(ious, conf_matrices, thresholds):
#                 if thres not in exit_to_thresh_to_details[exit_ix]:
#                     exit_to_thresh_to_details[exit_ix][thres] = {'iou': [],
#                                                                  'GT=fg,pred=fg': [],
#                                                                  'GT=bg,pred=bg': [],
#                                                                  'GT=fg,pred=bg': [],
#                                                                  'GT=bg,pred=fg': [],
#                                                                  'total': []}
#                 exit_to_thresh_to_details[exit_ix][thres]['iou'].append(iou)
#
#                 # GT: axis=1, pred: axis=0
#                 exit_to_thresh_to_details[exit_ix][thres]['GT=fg,pred=fg'].append(conf_matrix[1][1])
#                 exit_to_thresh_to_details[exit_ix][thres]['GT=bg,pred=bg'].append(conf_matrix[0][0])
#                 exit_to_thresh_to_details[exit_ix][thres]['GT=fg,pred=bg'].append(conf_matrix[0][1])
#                 exit_to_thresh_to_details[exit_ix][thres]['GT=bg,pred=fg'].append(conf_matrix[1][0])
#                 exit_to_thresh_to_details[exit_ix][thres]['total'].append(conf_matrix.sum())
#
#             exit_to_acc_metric[exit_ix].update(logits, batch['y'])
#
#         if batch_ix % save_every == 0:
#             logging.getLogger().info(f"Saving from batch #: {batch_ix}")
#             save_exitwise_heatmaps(batch['x'][0],
#                                    gt_masks[0],
#                                    {exit_ix: exit_to_gt_cams[exit_ix][0].squeeze() for exit_ix in exit_to_gt_cams},
#                                    save_dir=save_dir + f'/{_viz_ix}')
#             _viz_ix += 1
#
#     # Compute and print the metrics
#     exit_to_metrics = {}
#     for exit_ix in exit_to_thresh_to_details:
#         # peak_iou, peak_conf_matrix, peak_thresh = 0, 0, 0, 0
#         peak_vals = {'iou': 0}
#
#         for thresh in exit_to_thresh_to_details[exit_ix]:
#             m_iou = np.mean(exit_to_thresh_to_details[exit_ix][thresh]['iou'])
#             if m_iou > peak_vals['iou']:
#                 for k in exit_to_thresh_to_details[exit_ix][thresh]:
#                     peak_vals[k] = np.sum(exit_to_thresh_to_details[exit_ix][thresh][k]) / np.sum(
#                         exit_to_thresh_to_details[exit_ix][thresh]['total'])
#                 peak_vals['iou'] = np.mean(exit_to_thresh_to_details[exit_ix][thresh]['iou'])
#                 peak_vals['threshold'] = thresh
#
#         exit_to_metrics[exit_ix] = peak_vals
#         exit_to_metrics[exit_ix]['accuracy'] = exit_to_acc_metric[exit_ix].get_accuracy(),
#         exit_to_metrics[exit_ix]['mean_per_class'] = exit_to_acc_metric[exit_ix].get_mean_per_group_accuracy(
#             group_type='class')
#
#     print(json.dumps(exit_to_metrics, indent=4))
#     with open(os.path.join(save_dir, 'segmentation.json'), 'w') as f:
#         json.dump(exit_to_metrics, f, indent=4)


class SegmentationMetrics:
    def __init__(self):
        self.thresh_to_metrics = {}

    def update(self, gt_masks, pred_masks, h=None, w=None):
        """
        Computes iou, fg vs bg confusions and total pixels at different thresholds

        :param gt_masks: B x H_g x W_g or B x 3 x H_g x W_g
        :param pred_masks: B x H_p x W_p
        :return:
        """
        if len(gt_masks.shape) == 4:
            gt_masks = gt_masks.mean(dim=1)

        B, H_g, W_g = gt_masks.shape
        _, H_p, W_p = pred_masks.shape
        if H_g != H_p or W_g != W_p:
            # pred_masks = interpolate(pred_masks.unsqueeze(1), H_g, W_g).reshape(B, H_g, W_g)
            gt_masks = interpolate(gt_masks.unsqueeze(1), H_p, W_p).reshape(B, H_p, W_p)
        ious, conf_matrices, thresholds = get_binary_ious_conf_matrix(gt_masks.detach().cpu().long().flatten().numpy(),
                                                                      pred_masks.detach().cpu().flatten().numpy())
        for iou, conf_matrix, thresh in zip(ious, conf_matrices, thresholds):
            if thresh not in self.thresh_to_metrics:
                self.thresh_to_metrics[thresh] = {'iou': [],
                                                  'GT=fg,pred=fg': [],
                                                  'GT=bg,pred=bg': [],
                                                  'GT=fg,pred=bg': [],
                                                  'GT=bg,pred=fg': [],
                                                  'total': []}
            self.thresh_to_metrics[thresh]['iou'].append(iou)

            # GT: axis=1, pred: axis=0
            self.thresh_to_metrics[thresh]['GT=fg,pred=fg'].append(conf_matrix[1][1])
            self.thresh_to_metrics[thresh]['GT=bg,pred=bg'].append(conf_matrix[0][0])
            self.thresh_to_metrics[thresh]['GT=fg,pred=bg'].append(conf_matrix[0][1])
            self.thresh_to_metrics[thresh]['GT=bg,pred=fg'].append(conf_matrix[1][0])
            self.thresh_to_metrics[thresh]['total'].append(conf_matrix.sum())
        return self.thresh_to_metrics

    def summary(self):
        """
        Searches for peak iou among all the thresholds
        :return:
        """
        peak_vals = {'iou': -1}

        for thresh in self.thresh_to_metrics:
            m_iou = np.mean(self.thresh_to_metrics[thresh]['iou'])
            if m_iou > peak_vals['iou']:
                for k in self.thresh_to_metrics[thresh]:
                    # Normalize the confusion metrics
                    if 'GT=' in k:
                        peak_vals[k] = np.sum(self.thresh_to_metrics[thresh][k]) / np.sum(
                            self.thresh_to_metrics[thresh]['total'])
                peak_vals['iou'] = np.mean(self.thresh_to_metrics[thresh]['iou'])
                peak_vals['threshold'] = thresh
        return peak_vals


def save_exitwise_heatmaps(original, gt_mask, exit_to_heatmaps, save_dir, heat_map_suffix=''):
    """

    :param original: original image where the heatmaps will be placed
    :param gt_mask:
    :param exit_to_heatmaps:
    :param save_dir:
    :param heat_map_suffix:
    :return:
    """
    os.makedirs(save_dir, exist_ok=True)
    imwrite(os.path.join(save_dir, 'original.jpg'), to_numpy_img(original))
    ow, oh = original.shape[1], original.shape[2]
    original = (original - original.min()) / (original.max() - original.min())

    # Only show regions present in the gt mask
    _inter_mask = torch.relu(interpolate(gt_mask.unsqueeze(0).detach().cpu(), oh, ow) - 0.5).squeeze()
    imwrite(os.path.join(save_dir, 'true_mask.jpg'),
            to_numpy_img(original.detach().cpu() * _inter_mask)
            .squeeze())

    # Heat maps
    for exit_ix in exit_to_heatmaps:
        _exit_hm = compute_heatmap(original, exit_to_heatmaps[exit_ix])
        imwrite(os.path.join(save_dir, f'hm_exit_{exit_ix}{heat_map_suffix}.jpg'), _exit_hm)
        # print(f"Saved to {os.path.join(save_dir, f'hm_exit_{exit_ix}{heat_map_suffix}.jpg')}")


def save_heatmap(original, heatmap, save_dir, heat_map_suffix='', gt_mask=None):
    """

    :param original: original image where the heatmaps will be placed
    :param gt_mask:
    :param heatmap:
    :param save_dir:
    :param heat_map_suffix:
    :return:
    """
    os.makedirs(save_dir, exist_ok=True)
    imwrite(os.path.join(save_dir, 'original.jpg'), to_numpy_img(original))
    ow, oh = original.shape[1], original.shape[2]
    original = (original - original.min()) / (original.max() - original.min())

    # Only show regions present in the gt mask
    if gt_mask is not None:
        _inter_mask = torch.relu(interpolate(gt_mask.unsqueeze(0).detach().cpu(), oh, ow) - 0.5).squeeze()
        imwrite(os.path.join(save_dir, 'true_mask.jpg'),
                to_numpy_img(original.detach().cpu() * _inter_mask)
                .squeeze())

    # Heat maps
    _exit_hm = compute_heatmap(original, heatmap)
    imwrite(os.path.join(save_dir, f'hm{heat_map_suffix}.jpg'), _exit_hm)
    # print(f"Saved to {os.path.join(save_dir, f'hm{heat_map_suffix}.jpg')}")


def to_numpy_img(tensor):
    if len(tensor.shape) == 3:
        if tensor.shape[-1] != 3 and tensor.shape[0] == 3:
            tensor = torch.permute(tensor, (1, 2, 0))

    return (tensor.detach().cpu()).numpy()


def imwrite(save_file, img):
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    written = cv2.imwrite(save_file, (img * 255).astype(np.uint8))
    if not written:
        raise Exception(f'Could not write to {save_file}')
