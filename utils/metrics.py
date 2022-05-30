class Accuracy():
    """
    Tracks accuracy per class + group
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.correct_dict = {}
        self.total_dict = {}

    def update(self, pred_ys, gt_ys, class_names=None, group_names=None):
        for ix in range(len(pred_ys)):
            cls_name = str(gt_ys[ix]) if class_names is None else class_names[ix]
            grp_name = cls_name if group_names is None else group_names[ix]
            self.update_one(pred_ys[ix], gt_ys[ix], cls_name, 'class')
            self.update_one(pred_ys[ix], gt_ys[ix], grp_name, 'group')

    def update_one(self, pred_y, gt_y, name, grp_type):
        name = str(name)
        if grp_type not in self.total_dict:
            self.total_dict[grp_type] = {}
            self.correct_dict[grp_type] = {}

        if name not in self.total_dict[grp_type]:
            self.total_dict[grp_type][name] = 0
            self.correct_dict[grp_type][name] = 0

        if int(pred_y) == int(gt_y):
            self.correct_dict[grp_type][name] += 1
        self.total_dict[grp_type][name] += 1

    def get_per_group_accuracy(self, group_type, factor=1):
        assert group_type in self.total_dict
        per_group_accuracy = {}
        for group_name in self.correct_dict[group_type]:
            per_group_accuracy[group_name] \
                = self.correct_dict[group_type][group_name] / self.total_dict[group_type][group_name] * factor
        return per_group_accuracy

    def get_mean_per_group_accuracy(self, group_type, factor=1):
        total_acc, total_num = 0, 0
        per_group_accuracy = self.get_per_group_accuracy(group_type)
        for group_name in per_group_accuracy:
            total_acc += per_group_accuracy[group_name]
            total_num += 1
        return total_acc / total_num * factor

    def get_accuracy(self, group_type, factor=1):
        correct, total = 0, 0
        for grp_name in self.total_dict[group_type]:
            correct += self.correct_dict[group_type][grp_name]
            total += self.total_dict[group_type][grp_name]
        return correct / total * factor

    def summary(self, factor=100):
        obj = {}
        obj['accuracy'] = self.get_accuracy('class', factor)
        obj['MPC'] = self.get_mean_per_group_accuracy('class', factor)
        obj['MPG'] = self.get_mean_per_group_accuracy('group', factor)
        return obj

    def detailed(self, factor=100):
        obj = {}
        for group_type in self.total_dict:
            obj[group_type] = {
                'total': self.total_dict[group_type],
                'correct': self.correct_dict[group_type],
                'accuracy': self.get_accuracy(group_type, factor),
                'MPG': self.get_mean_per_group_accuracy(group_type, factor),
                'per_group': self.get_per_group_accuracy(group_type, factor)
            }
        return obj


class GateMetric():
    def __init__(self, threshold=0.5):
        self.total = 0
        self.exited = 0
        self.threshold = threshold

    def update(self, exit_probas):
        for proba in exit_probas:
            if proba >= self.threshold:
                self.exited += 1
            self.total += 1

    def get_exit_pct(self, factor=100):
        return self.exited / self.total * factor

    def summary(self):
        return {
            'Exit%': self.get_exit_pct()
        }
