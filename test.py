import pandas as pd


targets = pd.read_csv('processed/hBN_spin_500/targets.csv.gz').set_index('_id')
print(targets['spin_up_seq'])
