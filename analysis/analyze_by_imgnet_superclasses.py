import logging

from analysis.imgnet_hierarchy import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv

def compute_super_class_acc(logits, gt_labels, save_file=None):
    super_cls_to_total, super_cls_to_correct = {}, {}
    super_cls_to_acc = {}
    super_classes = get_super_classes(gt_labels)
    for super_cls, _logits, _y in zip(super_classes, logits, gt_labels):
        if super_cls not in super_cls_to_total:
            super_cls_to_total[super_cls] = 0
            super_cls_to_correct[super_cls] = 0
        super_cls_to_total[super_cls] += 1
        if int(torch.argmax(_logits)) == int(_y):
            super_cls_to_correct[super_cls] += 1

    for super_cls in super_cls_to_total:
        super_cls_to_acc[super_cls] = super_cls_to_correct[super_cls] / super_cls_to_total[super_cls] * 100
    if save_file is not None:
        with open(save_file+'.json', 'w') as f:
            json.dump(super_cls_to_acc, f, indent=4, sort_keys=True)

        with open(save_file+'.csv', 'w') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(list(super_cls_to_acc.keys())))
            writer.writeheader()
            writer.writerow(super_cls_to_acc)

    return super_cls_to_acc


def plot(model_to_super_cls_acc):
    data = {
        'Model': [],
        'Super Class': [],
        'Accuracy': []
    }

    for model in model_to_super_cls_acc:
        for super_cls in model_to_super_cls_acc[model]:
            data['Model'].append(model)
            data['Super Class'].append(super_cls)
            data['Accuracy'].append(model_to_super_cls_acc[model][super_cls] * 100)

    df = pd.DataFrame(data)
    chart = sns.barplot(data=df, x='Super Class', y='Accuracy', hue='Model')
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)

    plt.show()


if __name__ == "__main__":
    model_to_expt = {
        'ResNet50': '/home/robik/occam-networks-outputs/image_net/BaseTrainer/resnet50/subset_100/'
                    'preds_val_logits_epoch_90.pt',
        'OccamResNet50': '/home/robik/occam-networks-outputs/image_net/OccamTrainer/occam_resnet50/subset_100/'
                         'preds_val_early_logits_epoch_90.pt'
    }
    model_to_super_cls_acc = {}
    for model in model_to_expt:
        model_to_super_cls_acc[model] = compute_super_class_acc(model_to_expt[model])
    # print(json.dumps(model_to_super_cls_acc, indent=4, default=str, sort_keys=True))

    plot(model_to_super_cls_acc)
