import torch
import torch.nn as nn


class RNNHead(nn.Module):
    def __init__(self, seq_len, emb_size):
        super().__init__()
        self.gru_up = nn.GRU(input_size=2 * emb_size, hidden_size=2 * emb_size, batch_first=True)
        self.gru_down = nn.GRU(input_size=2 * emb_size, hidden_size=2 * emb_size, batch_first=True)
        self.seq_len = seq_len

    def forward(self, graph_embedding, embedded_seq, x_spin):
        x_up = x_spin[0]
        spin_up = x_spin[1]
        if spin_up:
            graph_embedding = torch.cat([graph_embedding, graph_embedding], dim=1)
        else:
            graph_embedding = torch.cat([graph_embedding, -graph_embedding], dim=1)

        return self.process_sequence(graph_embedding, embedded_seq, x_up)

    def process_sequence(self, graph_embedding, target_embedding, up):
        h_0 = graph_embedding[None, ...]
        m = self.gru_up if up else self.gru_down
        res, _ = m(target_embedding, h_0)
        return res

    def generate_sequence(self, graph_embedding, start_x, x_spin):
        x_up = x_spin[0]
        spin_up = x_spin[1]
        if spin_up:
            graph_embedding = torch.cat([graph_embedding, graph_embedding], dim=1)
        else:
            graph_embedding = torch.cat([graph_embedding, -graph_embedding], dim=1)

        res = []
        cur_h = graph_embedding[None, ...]
        cur_x = start_x
        m = self.gru_up if x_up else self.gru_down
        for i in range(self.seq_len):
            cur_x, cur_h = m(cur_x, cur_h)
            res.append(cur_x)
        return torch.cat(res, dim=1)


class RNNHeadAutoregressive(nn.Module):
    def __init__(self, seq_len, emb_size):
        super().__init__()
        self.gru_up = nn.GRU(input_size=2 * emb_size, hidden_size=2 * emb_size)
        self.gru_down = nn.GRU(input_size=2 * emb_size, hidden_size=2 * emb_size)
        self.seq_len = seq_len
        self.emb_size = emb_size

    def forward(self, x):
        x_spin_up = torch.cat([x, x], dim=1)
        x_spin_down = torch.cat([x, -x], dim=1)

        x_up_spin_up = self.process_sequence(x_spin_up, True)
        x_up_spin_down = self.process_sequence(x_spin_down, True)
        x_down_spin_up = self.process_sequence(x_spin_up, False)
        x_down_spin_down = self.process_sequence(x_spin_down, False)

        return x_up_spin_up, x_up_spin_down, x_down_spin_up, x_down_spin_down

    def process_sequence(self, x, up):
        res = []
        cur = x[None, ...]
        cur_h = torch.zeros((1, x.shape[0], 2 * self.emb_size))
        m = self.gru_up if up else self.gru_down
        for i in range(self.seq_len):
            cur, cur_h = m(cur, cur_h)
            res.append(cur.squeeze(0))
        return torch.stack(res, dim=2).permute(0, 2, 1)
