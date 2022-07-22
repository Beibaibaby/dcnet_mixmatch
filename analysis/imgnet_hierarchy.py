import torch
import json
import os
import sklearn.metrics as sk_metrics

import matplotlib.pyplot as plt

from analysis.imgnet_tree import imgnet_tree


# ImageNet hierarchy:
# https://gist.githubusercontent.com/mbostock/535395e279f5b83d732ea6e36932a9eb/raw/62863328f4841fce9452bc7e2f7b5e02b40f8f3d/mobilenet.json

def map_to_parents(node, child_to_parent={}, id_to_name={}, name_to_id={}):
    name = format_name(node['name'])
    id_to_name[node['id']] = name
    name_to_id[name] = node['id']
    if 'children' in node:
        for child in node['children']:
            child_id = child['id']
            if child_id not in child_to_parent:
                child_to_parent[child_id] = node['id']
                map_to_parents(child, child_to_parent, id_to_name, name_to_id)


def get_parent(id, child_to_parent, id_to_name,
               supercategories=['commodity', 'food', 'geological formation', 'natural object',
                                'organism', 'structure']):
    name = id_to_name[id]
    if name.lower() in ['french loaf', 'bun', 'carbonara', 'foodstuff', 'pretzel', 'sandwich', 'vegetable', 'feed',
                        'sauce', 'beverage']:
        return 'food'
    if name.lower() in ['plant', 'plant part', 'fungus']:
        return 'plant'
    if name.lower() in ['bag', 'basket', 'bin', 'box', 'can', 'case', 'envelope', 'glass', 'pot', 'package', 'shaker',
                        'savings bank']:
        return 'container'
    if name.lower() in ['bubble', 'connection', 'menu', 'street sign', 'traffic light', 'toilet tissue', 'dip',
                        'cassette', 'dish', 'dispenser', 'measure', 'receptacle', 'spoon', 'thimble']:
        return 'z_remaining'

    if id in child_to_parent:
        parent_id = child_to_parent[id]
        parent_name = id_to_name[parent_id]

        # if parent_name in ['instrumentality', 'container', 'device']:
        #     return name
        if parent_name in ['instrumentality', 'chordate', 'animal', 'organism', 'natural object', 'container',
                           "ImageNet 2011 Fall Release", 'vertebrate']:
            return name

    if name in supercategories:
        return name
    else:
        if id not in child_to_parent:
            return "z_remaining"
            # return id_to_name[id]
        else:
            return get_parent(child_to_parent[id], child_to_parent, id_to_name)


# def read_image_net_hierarchy(filepath='imagenet.json'):
#     with open(filepath) as f:
#         tree = json.load(f)
#     child_to_parent, id_to_name, name_to_id = {}, {}, {}
#     map_to_parents(tree, child_to_parent, id_to_name, name_to_id)
#     return child_to_parent, id_to_name, name_to_id

def read_image_net_hierarchy():
    # with open(filepath) as f:
    #     tree = json.load(f)
    child_to_parent, id_to_name, name_to_id = {}, {}, {}
    map_to_parents(imgnet_tree, child_to_parent, id_to_name, name_to_id)
    return child_to_parent, id_to_name, name_to_id


def get_image_net_class_ix_to_id(imgnet_dir='/home/robik/datasets/ImageNet1K/val'):
    names = list(sorted(os.listdir(imgnet_dir)))
    return {ix: name for ix, name in enumerate(names)}


def format_name(name):
    return name.split(",")[0]


def get_super_classes(labels, cls_ix_to_id=None, child_to_parent=None, id_to_name=None):
    if child_to_parent is None or id_to_name is None:
        child_to_parent, id_to_name, name_to_id = read_image_net_hierarchy()
    if cls_ix_to_id is None:
        cls_ix_to_id = get_image_net_class_ix_to_id()
    super_classes = []
    for lbl in labels:
        gt_id = cls_ix_to_id[int(lbl)]
        super_classes.append(get_parent(gt_id, child_to_parent, id_to_name))
    return super_classes


def get_gt_and_pred_names(logits, gt_labels, cls_ix_to_id, child_to_parent, id_to_name):
    pred_ys = torch.argmax(logits, dim=1)
    gt_names, pred_names = [], []
    for pred_ix, gt_ix in zip(pred_ys, gt_labels):
        gt_id = cls_ix_to_id[int(gt_ix)]
        pred_id = cls_ix_to_id[int(pred_ix)]
        gt_names.append(format_name(get_parent(gt_id, child_to_parent, id_to_name)))
        pred_names.append(format_name(get_parent(pred_id, child_to_parent, id_to_name)))
    return gt_names, pred_names


def get_stats_by_parents(gt_labels, cls_ix_to_id, child_to_parent, id_to_name):
    parent_to_cnt = {}
    parent_to_cls_names = {}
    for gt_ix in gt_labels:
        gt_id = cls_ix_to_id[int(gt_ix)]
        parent = get_parent(gt_id, child_to_parent, id_to_name)
        if parent not in parent_to_cnt:
            parent_to_cnt[parent] = 0
            parent_to_cls_names[parent] = {}

        parent_to_cnt[parent] += 1
        cls_name = id_to_name[gt_id]
        parent_to_cls_names[parent][cls_name] = cls_name
    return parent_to_cnt, parent_to_cls_names


def plot_conf_matrix(gt_names, pred_names, save_dir, save_fname='superclass_conf_matrix.png'):
    # plt.figure(figsize=(20, 20))
    fig, ax = plt.subplots(figsize=(15, 15))

    sk_metrics.ConfusionMatrixDisplay.from_predictions(gt_names, pred_names, normalize='true',
                                                       xticks_rotation='vertical', values_format='.0%', ax=ax)
    plt.savefig(os.path.join(save_dir, save_fname), bbox_inches='tight')
    print(f"Saved to {os.path.join(save_dir, save_fname)}")
    plt.close()


def load_output(file):
    preds = torch.load(file)
    return preds['logits'], preds['gt_labels']


if __name__ == "__main__":
    path = '/home/robik/occam-networks-outputs/image_net/OccamTrainer/occam_resnet18_group_norm_ws/subset_8'
    file = os.path.join(path, 'preds_Val_early_logits_epoch_90.pt')
    logits, gt_labels = load_output(file)
    child_to_parent, id_to_name, name_to_id = read_image_net_hierarchy()
    cls_ix_to_id = get_image_net_class_ix_to_id()
    parent_cnt, parent_to_cls_names = get_stats_by_parents(gt_labels, cls_ix_to_id, child_to_parent, id_to_name)
    print(json.dumps(parent_cnt, indent=4, sort_keys=True))
    print(json.dumps(parent_to_cls_names, indent=4, sort_keys=True))
    # gt_names, pred_names = get_gt_and_pred_names(logits, gt_labels, cls_ix_to_id, child_to_parent, id_to_name)
    # plot_conf_matrix(gt_names, pred_names, path)

# Super categories:
# commodity,  food, geological formation, instrumentality, natural object, organism, plaything, structure
# others:
# beverage, bubble, covering, creation, decoration, dip, fabric, feed, menu, padding, sauce, sheet, street sign, surface, toilet tissue, traffic light


# manually define
# {}
