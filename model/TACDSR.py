import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


class MLP(nn.Module):
    def __init__(self, hidden_dims, dropout=0.2):
        super().__init__()

        self.fc1 = nn.Linear(hidden_dims, 512)
        self.fc2 = nn.Linear(512, hidden_dims)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dims)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.layer_norm(x)
        return x


class Weight(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        nn.init.uniform_(self.mlp[-1].weight, -0.01, 0.01)
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x):
        x = self.mlp(x)
        return 0.3 * torch.sigmoid(x) + 0.3

class ATTENTION(torch.nn.Module):
    def __init__(self, opt):
        super(ATTENTION, self).__init__()
        self.opt = opt
        self.emb_dropout = torch.nn.Dropout(p=self.opt["dropout"])
        self.pos_emb = torch.nn.Embedding(self.opt["maxlen"], self.opt["hidden_units"], padding_idx=0)

        self.attention_layernorms = torch.nn.ModuleList()
        self.q_linears = torch.nn.ModuleList()
        self.k_linears = torch.nn.ModuleList()
        self.v_linears = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(self.opt["hidden_units"], eps=1e-8)

        self.gate = nn.Linear(self.opt["hidden_units"], self.opt["num_blocks"])

        for _ in range(self.opt["num_blocks"]):
            self.attention_layernorms.append(torch.nn.LayerNorm(self.opt["hidden_units"], eps=1e-8))

            self.q_linears.append(nn.Linear(self.opt["hidden_units"], self.opt["hidden_units"]))
            self.k_linears.append(nn.Linear(self.opt["hidden_units"], self.opt["hidden_units"]))
            self.v_linears.append(nn.Linear(self.opt["hidden_units"], self.opt["hidden_units"]))

            self.forward_layernorms.append(torch.nn.LayerNorm(self.opt["hidden_units"], eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(self.opt["hidden_units"],
                                                            self.opt["dropout"]))

    def forward(self, seqs_data, seqs, position, time_emb):
        seqs += self.pos_emb(position)
        time_emb += self.pos_emb(position)
        seqs = self.emb_dropout(seqs)
        time_emb = self.emb_dropout(time_emb)

        all_time_gates = torch.sigmoid(self.gate(time_emb))

        timeline_mask = torch.BoolTensor(seqs_data.cpu() == self.opt["itemnum"] - 1)
        if self.opt["cuda"]:
            timeline_mask = timeline_mask.cuda()
        seqs *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool))
        if self.opt["cuda"]:
            attention_mask = attention_mask.cuda()

        for i in range(self.opt["num_blocks"]):
            Q = self.attention_layernorms[i](seqs)

            q = self.q_linears[i](Q)
            k = self.k_linears[i](time_emb)
            v = self.v_linears[i](seqs + time_emb)

            batch_size, seq_len, _ = q.shape
            head_dim = self.opt["hidden_units"] // self.opt["num_heads"]
            q = q.view(batch_size, seq_len, self.opt["num_heads"], head_dim)
            k = k.view(batch_size, seq_len, self.opt["num_heads"], head_dim)
            v = v.view(batch_size, seq_len, self.opt["num_heads"], head_dim)

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)

            time_gate = all_time_gates[:, :, i:i + 1]
            time_gate_matrix = torch.matmul(time_gate, time_gate.transpose(1, 2))

            time_gate_matrix = time_gate_matrix.unsqueeze(1)
            gated_attn = attn_scores * time_gate_matrix

            gated_attn = gated_attn.masked_fill(attention_mask, float('-inf'))
            attn_weights = torch.softmax(gated_attn, dim=-1)
            output = torch.matmul(attn_weights, v)

            output = output.transpose(1, 2).contiguous()
            output = output.view(batch_size, seq_len, -1)

            seqs = Q + output

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        return self.last_layernorm(seqs)


class TACDSR(torch.nn.Module):
    def __init__(self, opt, adj, adj_single):
        super(TACDSR, self).__init__()
        self.opt = opt
        self.item_emb_X = torch.nn.Embedding(self.opt["itemnum"], self.opt["hidden_units"],
                                             padding_idx=self.opt["itemnum"] - 1)
        self.item_emb_Y = torch.nn.Embedding(self.opt["itemnum"], self.opt["hidden_units"],
                                             padding_idx=self.opt["itemnum"] - 1)
        self.item_emb = torch.nn.Embedding(self.opt["itemnum"], self.opt["hidden_units"],
                                           padding_idx=self.opt["itemnum"] - 1)

        self.time_emb = torch.nn.Embedding(self.opt["time_num"], self.opt["hidden_units"],
                                           padding_idx=0)

        print("max interval: ", int(self.opt["interval_scale"] * self.opt["max_interval"]))
        self.interval_time_emb = torch.nn.Embedding(int(self.opt["interval_scale"] * self.opt["max_interval"]) + 1, self.opt["hidden_units"],
                                           padding_idx=0)

        self.adj = adj
        self.adj_single = adj_single
        self.n_layers = 1

        self.lin_X = nn.Linear(self.opt["hidden_units"], self.opt["source_item_num"])
        self.lin_Y = nn.Linear(self.opt["hidden_units"], self.opt["target_item_num"])
        self.lin_PAD = nn.Linear(self.opt["hidden_units"], 1)
        self.encoder = ATTENTION(opt)
        self.encoder_X = ATTENTION(opt)
        self.encoder_Y = ATTENTION(opt)

        self.mlp_x = MLP(self.opt["hidden_units"])
        self.mlp_y = MLP(self.opt["hidden_units"])
        self.mlp = MLP(self.opt["hidden_units"])

        self.weight_x = Weight(64)
        self.weight_y = Weight(64)

    def graph_layer(self, adj_matrix, item_emb):
        side_emb = torch.sparse.mm(adj_matrix, item_emb)
        new_emb = F.dropout(side_emb, self.opt["dropout"], training=self.training)
        return new_emb

    def graph_transfer(self, item_emb_X, item_emb_Y, item_emb):
        item_emb = self.graph_layer(self.adj, item_emb)
        item_emb = F.normalize(item_emb, 2, -1)
        item_emb_X = self.graph_layer(self.adj_single, item_emb_X)
        item_emb_X = F.normalize(item_emb_X, 2, -1)
        item_emb_Y = self.graph_layer(self.adj_single, item_emb_Y)
        item_emb_Y = F.normalize(item_emb_Y, 2, -1)

        return item_emb_X, item_emb_Y, item_emb

    def graph_transfer_false(self, item_emb_X, item_emb_Y):
        item_emb_X = self.graph_layer(self.adj_single, item_emb_X)
        item_emb_X = F.normalize(item_emb_X, 2, -1)
        item_emb_Y = self.graph_layer(self.adj_single, item_emb_Y)
        item_emb_Y = F.normalize(item_emb_Y, 2, -1)

        return item_emb_X, item_emb_Y

    def forward(self, o_seqs, x_seqs, y_seqs, position, x_position, y_position, time, x_time, y_time, x_interval_time, y_interval_time, interval_time):
        x_seq = self.item_emb_X(x_seqs)
        y_seq = self.item_emb_Y(y_seqs)
        seq = self.item_emb(o_seqs)

        item_emb_X = self.item_emb_X.weight
        item_emb_Y = self.item_emb_Y.weight
        item_emb = self.item_emb.weight
        item_emb_X_list = [item_emb_X]
        item_emb_Y_list = [item_emb_Y]
        item_emb_list = [item_emb]

        for _ in range(self.n_layers):
            item_emb_X, item_emb_Y, item_emb = self.graph_transfer(item_emb_X, item_emb_Y, item_emb)
            item_emb_X_list = item_emb_X_list + [item_emb_X]
            item_emb_Y_list = item_emb_Y_list + [item_emb_Y]
            item_emb_list = item_emb_list + [item_emb]

        item_emb_X_list = torch.stack(item_emb_X_list, dim=1)
        item_emb_X = torch.mean(item_emb_X_list, dim=1, keepdim=False)
        item_emb_Y_list = torch.stack(item_emb_Y_list, dim=1)
        item_emb_Y = torch.mean(item_emb_Y_list, dim=1, keepdim=False)
        item_emb_list = torch.stack(item_emb_list, dim=1)
        item_emb = torch.mean(item_emb_list, dim=1, keepdim=False)

        x_interval = torch.log2(x_interval_time + 1)
        x_interval_index = torch.floor(self.opt["interval_scale"] * x_interval).long()
        x_interval_emb = self.interval_time_emb(x_interval_index)
        y_interval = torch.log2(y_interval_time + 1)
        y_interval_index = torch.floor(self.opt["interval_scale"] * y_interval).long()
        y_interval_emb = self.interval_time_emb(y_interval_index)
        interval = torch.log2(interval_time + 1)
        interval_index = torch.floor(self.opt["interval_scale"] * interval).long()
        interval_emb = self.interval_time_emb(interval_index)

        x_time_emb = self.mlp_x(x_interval_emb + self.time_emb(x_time))
        y_time_emb = self.mlp_y(y_interval_emb + self.time_emb(y_time))
        time_emb = self.mlp(interval_emb + self.time_emb(time))

        seqs = item_emb[o_seqs] + seq
        seqs *= self.item_emb.embedding_dim ** 0.5
        seqs_fea = self.encoder(o_seqs, seqs, position, time_emb)

        seqs = item_emb_X[x_seqs] + x_seq
        seqs *= self.item_emb.embedding_dim ** 0.5
        x_seqs_fea = self.encoder_X(x_seqs, seqs, x_position, x_time_emb)

        seqs = item_emb_Y[y_seqs] + y_seq
        seqs *= self.item_emb.embedding_dim ** 0.5
        y_seqs_fea = self.encoder_Y(y_seqs, seqs, y_position, y_time_emb)

        item_emb_X = self.item_emb_X.weight[:self.opt['source_item_num'], :]
        item_emb_Y = self.item_emb_Y.weight[self.opt['source_item_num']:self.opt['source_item_num'] + self.opt["target_item_num"], :]

        return seqs_fea, x_seqs_fea, y_seqs_fea, item_emb_X, item_emb_Y

    def false_forward(self, x_aug, y_aug, x_aug_pos, y_aug_pos, x_aug_time, y_aug_time, x_aug_interval, y_aug_interval):
        x_seq = self.item_emb_X(x_aug)
        y_seq = self.item_emb_Y(y_aug)

        item_emb_X = self.item_emb_X.weight
        item_emb_Y = self.item_emb_Y.weight
        item_emb_X_list = [item_emb_X]
        item_emb_Y_list = [item_emb_Y]

        for _ in range(self.n_layers):
            item_emb_X, item_emb_Y = self.graph_transfer_false(item_emb_X, item_emb_Y)
            item_emb_X_list = item_emb_X_list + [item_emb_X]
            item_emb_Y_list = item_emb_Y_list + [item_emb_Y]

        item_emb_X_list = torch.stack(item_emb_X_list, dim=1)
        item_emb_X = torch.mean(item_emb_X_list, dim=1, keepdim=False)
        item_emb_Y_list = torch.stack(item_emb_Y_list, dim=1)
        item_emb_Y = torch.mean(item_emb_Y_list, dim=1, keepdim=False)

        x_interval = torch.log2(x_aug_interval + 1)
        x_interval_index = torch.floor(self.opt["interval_scale"] * x_interval).long()
        x_interval_emb = self.interval_time_emb(x_interval_index)
        y_interval = torch.log2(y_aug_interval + 1)
        y_interval_index = torch.floor(self.opt["interval_scale"] * y_interval).long()
        y_interval_emb = self.interval_time_emb(y_interval_index)

        x_time_emb = self.mlp_x(x_interval_emb + self.time_emb(x_aug_time))
        y_time_emb = self.mlp_y(y_interval_emb + self.time_emb(y_aug_time))

        seqs = item_emb_X[x_aug] + x_seq
        seqs *= self.item_emb.embedding_dim ** 0.5
        x_seqs_fea = self.encoder_X(x_aug, seqs, x_aug_pos, x_time_emb)

        seqs = item_emb_Y[y_aug] + y_seq
        seqs *= self.item_emb.embedding_dim ** 0.5
        y_seqs_fea = self.encoder_Y(y_aug, seqs, y_aug_pos, y_time_emb)

        return x_seqs_fea, y_seqs_fea
