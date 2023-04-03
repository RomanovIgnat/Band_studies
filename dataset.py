import pandas as pd
from bisect import bisect_left

import torch

folds = pd.read_csv('experiments/hBN_test_exp_test/folds.csv.gz', index_col="_id")
folds = folds.loc[:, 'fold']

targets = pd.read_csv('processed/hBN_spin_500/targets.csv.gz').set_index('_id')
targets = targets[['spin_up_seq', 'spin_down_seq', 'fermi_level']]

data = pd.read_pickle('processed/hBN_spin_500/data.pickle.gz')['defect_representation']


def process_energy_levels(edata):
    res = []
    for row in edata:
        efermi = row[2]
        spin_up_seq = eval(row[0])
        spin_down_seq = eval(row[1])
        i = bisect_left(spin_up_seq, efermi)
        j = bisect_left(spin_down_seq, efermi)
        res.append([torch.Tensor(list(reversed(spin_up_seq[i - 5: i])))[None, :],
                    torch.Tensor(spin_down_seq[i: i + 5])[None, :],
                    torch.Tensor(list(reversed(spin_down_seq[j - 5: j])))[None, :],
                    torch.Tensor(spin_down_seq[j: j + 5])[None, :],
                    torch.Tensor([efermi])[None, :]])
    return res


train_folds = {0, 2}
train_ids = folds[folds.isin(train_folds)]
train = data.reindex(index=train_ids.index)
train_targets = targets.reindex(index=train_ids.index)
train_targets = train_targets.values.tolist()
train_targets = process_energy_levels(train_targets)

test_ids = folds[folds == 1]
test = data.reindex(index=test_ids.index)
test_targets = targets.reindex(index=test_ids.index)
test_targets = test_targets.values.tolist()
test_targets = process_energy_levels(test_targets)


def set_attr(structure, attrs, names):
    for attr, name in zip(attrs, names):
        setattr(structure, name, attr)
    return structure


train_data = [set_attr(s, y, ['spin_up_down',
                              'spin_up_up',
                              'spin_down_down',
                              'spin_down_up',
                              'fermi_level']) for s, y in zip(train, train_targets)]
test_data = [set_attr(s, y, ['spin_up_down',
                             'spin_up_up',
                             'spin_down_down',
                             'spin_down_up',
                             'fermi_level']) for s, y in zip(test, test_targets)]
