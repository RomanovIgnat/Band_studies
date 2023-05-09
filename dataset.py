import pandas as pd
from bisect import bisect_left

import torch

folds = pd.read_csv('experiments/InSe_test_exp_test/folds.csv.gz', index_col="_id")
folds = folds.loc[:, 'fold']

targets = pd.read_csv('processed/InSe_spin_500/targets.csv.gz').set_index('_id')
targets = targets[['spin_up_seq', 'spin_down_seq', 'fermi_level']]

data = pd.read_pickle('processed/InSe_spin_500/data.pickle.gz')['defect_representation']


def process_energy_levels(edata):
    res = []
    for row in edata:
        efermi = row[2]
        spin_up_seq = eval(row[0])
        spin_down_seq = eval(row[1])
        spin_up_seq = [level - efermi for level in spin_up_seq]
        spin_down_seq = [level - efermi for level in spin_down_seq]
        i = bisect_left(spin_up_seq, 0)
        j = bisect_left(spin_down_seq, 0)
        res.append([torch.tensor(list(reversed(spin_up_seq[i - 10: i])))[None, :],
                    torch.tensor(spin_down_seq[i: i + 10])[None, :],
                    torch.tensor(list(reversed(spin_down_seq[j - 10: j])))[None, :],
                    torch.tensor(spin_down_seq[j: j + 10])[None, :]
                    ])
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
                              ]) for s, y in zip(train, train_targets)]

test_data = [set_attr(s, y, ['spin_up_down',
                             'spin_up_up',
                             'spin_down_down',
                             'spin_down_up',
                             ]) for s, y in zip(test, test_targets)]
