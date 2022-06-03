import numpy as np
import torch
import cv2
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def cams_to_masks(cams, threshold=None):
    """
    Generates binary masks by applying the threshold in the CAMs
    :param cams: B x H x W
    :param threshold: Default: uses mean CAM value per sample as the threshold
    :return:
    """
    cam_scores = torch.sigmoid(cams)
    if threshold is None:
        threshold = cam_scores.mean(dim=2).mean(dim=1).unsqueeze(1).unsqueeze(2).repeat(1, cams.shape[1],
                                                                                        cams.shape[2]).detach()
    return (cam_scores >= threshold).float()


def binary_mask_to_bbox(binary_mask, margin=0):
    """
    Obtains bounding box enclosing the activated regions in the binary_mask

    :param binary_mask:
    :param margin:
    :return:
    """
    if isinstance(binary_mask, torch.Tensor):
        binary_mask = binary_mask.detach().cpu().numpy()
    nonzero_idx = np.nonzero(binary_mask)

    pt1, pt2 = [0] * binary_mask.ndim, [0] * binary_mask.ndim
    for i in range(binary_mask.ndim):
        pt1[i] = max(0, np.min(nonzero_idx[i]) - margin)
        pt2[i] = min(binary_mask.shape[i], np.max(nonzero_idx[i]) + margin + 1)

    return list(reversed(pt1)), list(reversed(pt2))


def np_crop_and_resize(masks, bboxes, size):
    cropped_bmasks = [mask.float()[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]].detach().cpu().numpy() for
                      mask, bbox in zip(masks, bboxes)]
    if size is not None:
        cropped_bmasks = [cv2.resize(mask, size, interpolation=cv2.INTER_LINEAR) for mask in cropped_bmasks]
    return cropped_bmasks


def torch_crop_and_resize(cams, bboxes, size):
    """

    :param cams: B x H x W
    :param bboxes: 1 Bounding box per CAM used to extract the shape
    :param size: Cropped CAMs are resized to this size
    :return:
    """
    cropped_cams = [cam[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] for cam, bbox in zip(cams, bboxes)]
    cropped_cams = [
        F.interpolate(cam.unsqueeze(0).unsqueeze(1), size=size, align_corners=False, mode='bilinear').squeeze() for
        cam in cropped_cams]
    cropped_cams = torch.stack(cropped_cams, dim=0)
    return cropped_cams


def interpolate(x, h, w):
    if len(x.shape) == 2:
        x = x.unsqueeze(0).unsqueeze(1)
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    if x.shape[2] == h and x.shape[3] == w:
        return x
    return F.interpolate(x, (h, w), mode='bilinear', align_corners=False).squeeze()


def get_early_exit_cams(model_out, H=None, W=None):
    """
    Returns the CAMs from the early exits
    :param model_out: Should contain cams from different exits named as 'exit_name, cam' and
    early_exit_names specifying the name of early exit for each sample
    :param H:
    :param W:
    :return:
    """
    ee_names = model_out['early_exit_names']
    cam_dict = {}
    for k in model_out.keys():
        if 'cam' in k:
            cam_dict[k] = interpolate(model_out[k].detach().cpu(), H, W)

    ee_cams = torch.zeros((len(ee_names), model_out[k].shape[1], H, W))
    for ix, ee_name in enumerate(ee_names):
        cam_key = f"{ee_name}, cam"
        ee_cams[ix] = cam_dict[cam_key][ix]
    return ee_cams


def get_gt_class_cams(model, batch, device=None, target_exit_size=1):
    """
    Returns CAMs for OccamNets and GradCAMs for other models for ground truth classes
    :param model:
    :param batch: should contain 'x' and 'y'
    :param target_exit_size: CAMs are resized to this exit's spatial dims
    :return:
    """
    if device is not None:
        batch['x'] = batch['x'].to(device)

    if 'occam' in type(model).__name__.lower():
        model_out = model(batch['x'])
        exit_to_gt_cams = {}
        resize_H, resize_W = model_out[f"E={target_exit_size}, cam"].shape[2], \
                             model_out[f"E={target_exit_size}, cam"].shape[3]

        # Get GT class CAMs for each exit
        for exit_ix in range(len(model.multi_exit.get_exit_block_nums())):
            model_out[f"E={exit_ix}, cam"] = interpolate(model_out[f"E={exit_ix}, cam"], resize_H, resize_W)
            cams = model_out[f"E={exit_ix}, cam"]
            gt_ys_ixs = batch['y'].squeeze().unsqueeze(1).unsqueeze(2).unsqueeze(3) \
                .repeat(1, 1, cams.shape[2], cams.shape[3])
            if device is not None:
                gt_ys_ixs = gt_ys_ixs.to(device)
            gt_cams = cams.gather(1, gt_ys_ixs).squeeze()
            exit_to_gt_cams[exit_ix] = gt_cams

        # Obtain early exit cams
        ee_cams = get_early_exit_cams(model_out, resize_H, resize_W)
        model_out["E=early_exit, cam"] = ee_cams
        gt_ys_ixs = batch['y'].squeeze().unsqueeze(1).unsqueeze(2).unsqueeze(3) \
            .repeat(1, 1, ee_cams.shape[2], ee_cams.shape[3])
        if device is not None:
            gt_ys_ixs = gt_ys_ixs.to(device)

        ee_gt_cams = ee_cams.gather(1, gt_ys_ixs.detach().cpu()).squeeze()
        exit_to_gt_cams['early_exit'] = ee_gt_cams

        return exit_to_gt_cams, model_out
    else:
        model_out = model(batch['x'])
        # Get GradCAMs
        grad_cam = GradCAM(model=model, target_layers=get_target_layers(model))
        targets = [ClassifierOutputTarget(int(y)) for y in batch['y']]
        target_cams = grad_cam(input_tensor=batch['x'], targets=targets)
        return {
                   0: torch.from_numpy(target_cams)
               }, model_out


def get_target_layers(model):
    """
    Get convolutional layer for GradCAM
    :param model:
    :return:
    """
    if 'VariableWidthResNet' in type(model).__name__:
        return [model.layer4[-1]]
    else:
        raise Exception(f"Specify the target layer for {type(model)}")
