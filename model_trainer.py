import numpy as np
import torch.nn.functional as F
import torch.optim
from joblib import Parallel, delayed
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from model import MEGNet
from struct2graph import FlattenGaussianDistanceConverter, AtomFeaturesExtractor, SimpleCrystalConverter
from utils import Scaler
from dataset import train_data, test_data


bond_converter = FlattenGaussianDistanceConverter(centers=np.linspace(0, 10, 10))
atom_converter = AtomFeaturesExtractor('Z')
converter = SimpleCrystalConverter(
    bond_converter=bond_converter,
    atom_converter=atom_converter,
    cutoff=10,
    add_z_bond_coord=True,
    add_eos_features=True,
    target_names=['spin_up_down', 'spin_up_up', 'spin_down_down', 'spin_down_up', 'fermi_level']
)
model = MEGNet(
    edge_input_shape=bond_converter.get_shape(eos=True),
    node_input_shape=atom_converter.get_shape(),
    embedding_size=32,
    n_blocks=2,
    state_input_shape=2,
    vertex_aggregation='mean',
    global_aggregation='mean',
    seq_len=5
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scaler = Scaler()

train_dataset = train_data
test_dataset = test_data

print("converting data")
train_structures = Parallel(n_jobs=2, backend='threading')(
    delayed(converter.convert)(s) for s in tqdm(train_dataset))
test_structures = Parallel(n_jobs=2, backend='threading')(
    delayed(converter.convert)(s) for s in tqdm(test_dataset))
scaler.fit(train_structures, ['spin_up_down',
                              'spin_up_up',
                              'spin_down_down',
                              'spin_down_up',
                              'fermi_level'])

trainloader = DataLoader(train_structures, batch_size=64, shuffle=True, num_workers=0)
testloader = DataLoader(test_structures, batch_size=50, shuffle=False, num_workers=0)

for epoch in range(100):
    model.train(True)
    for i, batch in tqdm(enumerate(trainloader)):
        # batch = batch.to(self.device)
        x_up_spin_up, x_up_spin_down, x_down_spin_up, x_down_spin_down, fermi = model(
            batch.x, batch.edge_index, batch.edge_attr, batch.state, batch.batch, batch.bond_batch
        )

        loss = F.mse_loss(x_up_spin_up, scaler.transform(batch.spin_up_up)) + \
            F.mse_loss(x_up_spin_down, scaler.transform(batch.spin_down_up)) + \
            F.mse_loss(x_down_spin_up, scaler.transform(batch.spin_up_down)) + \
            F.mse_loss(x_down_spin_down, scaler.transform(batch.spin_down_down)) + \
            F.mse_loss(fermi, scaler.transform(batch.fermi_level))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for batch in testloader:
        x_up_spin_up, x_up_spin_down, x_down_spin_up, x_down_spin_down, fermi = model(
            batch.x, batch.edge_index, batch.edge_attr, batch.state, batch.batch, batch.bond_batch
        )

        mae = F.l1_loss(scaler.inverse_transform(x_up_spin_up), batch.spin_up_up) + \
            F.l1_loss(scaler.inverse_transform(x_up_spin_down), batch.spin_down_up) + \
            F.l1_loss(scaler.inverse_transform(x_down_spin_up), batch.spin_up_down) + \
            F.l1_loss(scaler.inverse_transform(x_down_spin_down), batch.spin_down_down) + \
            F.l1_loss(scaler.inverse_transform(fermi), batch.fermi_level)

        print(batch.spin_up_up[0])
        print(scaler.inverse_transform(x_up_spin_up)[0])
        print(epoch, mae / 5)
