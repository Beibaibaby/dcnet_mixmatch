import os
import numpy as np
import torch
import cv2
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from utils.data_utils import get_dir


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


def sigmoid(x):
    return torch.sigmoid(x)


def min_max_norm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-5)


def compute_heatmap(img, cam, norm_type=min_max_norm):
    # Code from: Core risk minimization
    assert len(img.shape) == 3
    if img.shape[0] == 3 and img.shape[-1] != 3:
        img = torch.permute(img, (1, 2, 0))

    assert len(cam.shape) == 2  # Expects a grayscale CAM
    if cam.shape[0] != img.shape[0] or cam.shape[1] != img.shape[1]:
        cam = interpolate(cam, img.shape[0], img.shape[1])
    hm = cv2.applyColorMap(np.uint8(255 * norm_type(cam).detach().cpu().numpy()), cv2.COLORMAP_JET)
    hm = np.float32(hm) / 255
    hm = img.detach().cpu().numpy() + hm
    return hm / np.max(hm)


def cosine_similarity(tensor1, tensor2):
    """
    Measures similarity between each cell in tensor1 with every other cell of the same sample in tensor2
    :param tensor1: B x D x HW
    :param tensor2: B x D x k
    :return: Cosine similarity between each HW with each k
    """
    assert tensor1.shape[1] == tensor2.shape[1]
    cosine = F.normalize(tensor1, dim=1).permute(0, 2, 1) @ F.normalize(tensor2, dim=1)
    return cosine.sum(dim=2)


def get_class_cams_for_occam_nets(cams, classes):
    """
    Returns CAMs for OccamNets and GradCAMs for other models for given classes
    :param cams: B x C x H x W
    :param classes: gathers CAM for these classes
    :return:
    """
    if isinstance(classes, list):
        classes = torch.LongTensor(classes, device=cams.device)
    if classes.device != cams.device:
        classes = classes.to(cams.device)

    _classes = classes.squeeze().unsqueeze(1).unsqueeze(2).unsqueeze(3) \
        .repeat(1, 1, cams.shape[2], cams.shape[3])
    class_cams = cams.gather(1, _classes).squeeze()
    return class_cams


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


def get_class_cams(x, model, classes):
    grad_cam = GradCAM(model=model, target_layers=get_target_layers(model))
    targets = [ClassifierOutputTarget(int(y)) for y in classes]
    return torch.from_numpy(grad_cam(input_tensor=x, targets=targets))


def imwrite(save_file, img):
    dir = get_dir(save_file)
    os.makedirs(dir, exist_ok=True)
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    written = cv2.imwrite(save_file, (img * 255).astype(np.uint8))
    if not written:
        raise Exception(f'Could not write to {save_file}')
