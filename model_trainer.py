import random
import numpy as np
import torch.nn.functional as F
import torch.optim
import wandb
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from torch_geometric.loader import DataLoader
from tqdm import trange, tqdm

from model import MEGNet
from struct2graph import FlattenGaussianDistanceConverter, AtomFeaturesExtractor, SimpleCrystalConverter
from utils import Scaler, InvertibleEmbedder
from dataset import train_data, test_data
from rnn_head import RNNHead


seed = 17
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

bond_converter = FlattenGaussianDistanceConverter(centers=np.linspace(0, 10, 10))
atom_converter = AtomFeaturesExtractor('werespecies')
converter = SimpleCrystalConverter(
    bond_converter=bond_converter,
    atom_converter=atom_converter,
    cutoff=10,
    add_z_bond_coord=True,
    add_eos_features=False,
    target_names=['spin_up_down', 'spin_up_up', 'spin_down_down', 'spin_down_up']
)
model = MEGNet(
    edge_input_shape=bond_converter.get_shape(eos=False),
    node_input_shape=atom_converter.get_shape(),
    embedding_size=32,
    n_blocks=2,
    state_input_shape=2,
    vertex_aggregation='sum',
    global_aggregation='sum',
    seq_len=10,
)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
# scaler = Scaler()

train_dataset = train_data
test_dataset = test_data

print("converting data")
train_structures = Parallel(n_jobs=2, backend='threading')(
    delayed(converter.convert)(s) for s in tqdm(train_dataset))
test_structures = Parallel(n_jobs=2, backend='threading')(
    delayed(converter.convert)(s) for s in tqdm(test_dataset))
# scaler.fit(train_structures, ['spin_up_down',
#                               'spin_up_up',
#                               'spin_down_down',
#                               'spin_down_up',
#                               ])

trainloader = DataLoader(train_structures, batch_size=64, shuffle=True, num_workers=0)
testloader = DataLoader(test_structures, batch_size=50, shuffle=False, num_workers=0)

wandb.init(
    project="band_structure",
    entity='ignat'
)

for epoch in trange(1000):
    if epoch == 300:
        optimizer.param_groups[0]['lr'] *= 0.8
    losses = []
    maes = []
    model.train(True)
    for i, batch in enumerate(trainloader):
        # batch = batch.to(self.device)
        x_up_spin_up, x_up_spin_down, x_down_spin_up, x_down_spin_down, fermi = model(
            batch.x, batch.edge_index, batch.edge_attr, batch.state, batch.batch, batch.bond_batch
        )

        loss = F.mse_loss(x_up_spin_up, batch.spin_up_up) + \
               F.mse_loss(x_up_spin_down, batch.spin_down_up) + \
               F.mse_loss(x_down_spin_up, batch.spin_up_down) + \
               F.mse_loss(x_down_spin_down, batch.spin_down_down) + \
               F.mse_loss(fermi, torch.zeros_like(fermi))

        mae = F.l1_loss(x_up_spin_up, batch.spin_up_up, reduction='mean') + \
              F.l1_loss(x_up_spin_down, batch.spin_down_up, reduction='mean') + \
              F.l1_loss(x_down_spin_up, batch.spin_up_down, reduction='mean') + \
              F.l1_loss(x_down_spin_down, batch.spin_down_down, reduction='mean')

        losses.append(loss.data.numpy())
        maes.append((mae / 4).data.numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.train(False)
    with torch.no_grad():
        for batch in testloader:
            x_up_spin_up, x_up_spin_down, x_down_spin_up, x_down_spin_down, _ = model(
                batch.x, batch.edge_index, batch.edge_attr, batch.state, batch.batch, batch.bond_batch
            )

            mae = F.l1_loss(x_up_spin_up, batch.spin_up_up, reduction='sum') + \
                  F.l1_loss(x_up_spin_down, batch.spin_down_up, reduction='sum') + \
                  F.l1_loss(x_down_spin_up, batch.spin_up_down, reduction='sum') + \
                  F.l1_loss(x_down_spin_down, batch.spin_down_down, reduction='sum')

            fig = plt.figure(figsize=(15, 10))
            for i in range(5):
                plt.subplot(2, 5, i + 1)
                plt.hlines(
                    x_up_spin_up[10 * i].data.numpy().tolist() + \
                    x_down_spin_up[10 * i].data.numpy().tolist(),
                    0, 1,
                    colors='g',
                    linestyles='solid'
                )
                plt.hlines(
                    batch.spin_up_up[10 * i].data.numpy().tolist() + \
                    batch.spin_up_down[10 * i].data.numpy().tolist(),
                    0, 1,
                    colors='r',
                    linestyles='dashed'
                )
                plt.subplot(2, 5, i + 6)
                plt.hlines(
                    x_up_spin_down[10 * i].data.numpy().tolist() + \
                    x_down_spin_down[10 * i].data.numpy().tolist(),
                    0, 1,
                    colors='b',
                    linestyles='solid'
                )
                plt.hlines(
                    batch.spin_down_up[10 * i].data.numpy().tolist() + \
                    batch.spin_down_down[10 * i].data.numpy().tolist(),
                    0, 1,
                    colors='tab:orange',
                    linestyles='dashed'
                )

            wandb.log({
                'train loss': np.mean(losses),
                'test mae': mae.data.numpy() / (50 * 10 * 4),
                'train mae': np.mean(maes),
                'diagrams': wandb.Image(fig)
            })
            plt.close(fig)

#torch.save(model.state_dict(), 'model.pt')
