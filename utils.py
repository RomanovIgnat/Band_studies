import numpy as np
from copy import copy
import torch


class Scaler:
    def __init__(self):
        self.mean = 0
        self.std = 1.0

    def fit(self, dataset, feature_names):
        data = np.concatenate([
            getattr(dataset[i], name).squeeze(0).data.numpy() for name in feature_names for i in range(len(dataset))
        ])
        self.mean = np.mean(data)
        self.std = np.std(data)

    def transform(self, data):
        data_copy = copy(data)
        return (data_copy - self.mean) / (self.std if abs(self.std) > 1e-7 else 1.)

    def inverse_transform(self, data):
        data_copy = data.detach().clone()
        std = self.std if abs(self.std) > 1e-7 else 1.0
        return data_copy * std + self.mean


class InvertibleEmbedder:
    def __init__(self, centers, sigma):
        self.centers = centers
        self.sigma = sigma

    def embed(self, x):
        # x - batch * seq_len
        return torch.exp(
            -((x[..., None] - self.centers) / self.sigma) ** 2
        )

    def solve(self, embs, tmp):
        solution = torch.mean(tmp + self.centers, dim=-1)
        err = ((self.embed(solution) - embs) ** 2).sum(dim=-1)
        return solution, err

    def inverse(self, embs):
        # embs batch * seq_len * emb_size
        tmp = torch.sqrt(-torch.log(embs.clamp(1e-15, 1))) * self.sigma
        max_idx = torch.argmax(embs, dim=-1, keepdim=True)
        tmp1 = torch.where(torch.arange(tmp.shape[-1]) < max_idx, tmp, -tmp)
        tmp2 = torch.where(torch.arange(tmp.shape[-1]) <= max_idx, tmp, -tmp)
        solution1, err1 = self.solve(embs, tmp1)
        solution2, err2 = self.solve(embs, tmp2)
        return torch.where(err1 < err2, solution1, solution2)


if __name__ == '__main__':
    embedder = InvertibleEmbedder(torch.linspace(-2, 2, 30), 0.5)
    batch = torch.tensor([[-1., 1.], [-0.5, 0.3]])
    print(embedder.inverse(embedder.embed(batch)))
