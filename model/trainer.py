import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import torch_utils
from model.TACDSR import TACDSR
from model.diffusion import TimeDiff
from model.utils import *
import pdb
import numpy as np
import time as tm


class Trainer(object):
    def __init__(self, opt):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):  # here should change
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename, epoch):
        params = {
            'model': self.model.state_dict(),
            'config': self.opt,
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")


class CDSRTrainer(Trainer):
    def __init__(self, opt, adj=None, adj_single=None):
        self.opt = opt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TACDSR(opt, adj, adj_single)
        self.Diffusion = TimeDiff(opt, self.device, opt['mean_type'], opt['noise_schedule'], opt['noise_scale'],
                                  opt['noise_max'], opt['noise_min'], opt['steps'], opt['beta_fixed'], opt['embedding_size'],
                                  opt['emb_size'], opt['hid_temp'], opt['norm'], opt['reweight'], opt['sampling_steps'],
                                  opt['sampling_noise'], opt['mlp_act_func'], opt['history_num_per_term'], opt['dims_dnn'])

        self.mi_loss = 0
        self.BCE_criterion = nn.BCEWithLogitsLoss()
        self.CS_criterion = nn.CrossEntropyLoss(reduction='none')
        if opt['cuda']:
            self.model.cuda()
            self.Diffusion.cuda()
            self.BCE_criterion.cuda()
            self.CS_criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.model.parameters(), opt['lr'])
        self.optimizer2 = torch_utils.get_optimizer(opt['optim'], self.Diffusion.parameters(), opt['lr'])
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()

    def unpack_batch_predict(self, batch):
        if self.opt["cuda"]:
            inputs = [Variable(b.cuda()) for b in batch]
            seq = inputs[0]
            x_seq = inputs[1]
            y_seq = inputs[2]
            position = inputs[3]
            x_position = inputs[4]
            y_position = inputs[5]
            X_last = inputs[6]
            Y_last = inputs[7]
            x_ground = inputs[8]
            y_ground = inputs[9]
            negative_sample_x = inputs[10]
            negative_sample_y = inputs[11]
            time = inputs[12]
            x_time = inputs[13]
            y_time = inputs[14]
            x_time_window = inputs[15]
            y_time_window = inputs[16]
            len_x = inputs[17]
            len_y = inputs[18]
            x_interval_time = inputs[19]
            y_interval_time = inputs[20]
            interval_time = inputs[21]
        else:
            inputs = [Variable(b) for b in batch]
            seq = inputs[0]
            x_seq = inputs[1]
            y_seq = inputs[2]
            position = inputs[3]
            x_position = inputs[4]
            y_position = inputs[5]
            X_last = inputs[6]
            Y_last = inputs[7]
            x_ground = inputs[8]
            y_ground = inputs[9]
            negative_sample_x = inputs[10]
            negative_sample_y = inputs[11]
            time = inputs[12]
            x_time = inputs[13]
            y_time = inputs[14]
            x_time_window = inputs[15]
            y_time_window = inputs[16]
            len_x = inputs[17]
            len_y = inputs[18]
            x_interval_time = inputs[19]
            y_interval_time = inputs[20]
            interval_time = inputs[21]
        return seq, x_seq, y_seq, position, x_position, y_position, X_last, Y_last, x_ground, y_ground, negative_sample_x, negative_sample_y, time, x_time, y_time, x_time_window, y_time_window, len_x, len_y, x_interval_time, y_interval_time, interval_time

    def unpack_batch(self, batch):
        if self.opt["cuda"]:
            inputs = [Variable(b.cuda()) for b in batch]
            seq = inputs[0]
            x_seq = inputs[1]
            y_seq = inputs[2]
            position = inputs[3]
            x_position = inputs[4]
            y_position = inputs[5]
            ground = inputs[6]
            share_x_ground = inputs[7]
            share_y_ground = inputs[8]
            x_ground = inputs[9]
            y_ground = inputs[10]
            ground_mask = inputs[11]
            share_x_ground_mask = inputs[12]
            share_y_ground_mask = inputs[13]
            x_ground_mask = inputs[14]
            y_ground_mask = inputs[15]
            time = inputs[16]
            x_time = inputs[17]
            y_time = inputs[18]
            x_time_window = inputs[19]
            y_time_window = inputs[20]
            len_x = inputs[21]
            len_y = inputs[22]
            x_ground_mask_share = inputs[23]
            y_ground_mask_share = inputs[24]
            a_position = inputs[25]
            x_interval_time = inputs[26]
            y_interval_time = inputs[27]
            interval_time = inputs[28]
            raw_interval_time = inputs[29]
        else:
            inputs = [Variable(b) for b in batch]
            seq = inputs[0]
            x_seq = inputs[1]
            y_seq = inputs[2]
            position = inputs[3]
            x_position = inputs[4]
            y_position = inputs[5]
            ground = inputs[6]
            share_x_ground = inputs[7]
            share_y_ground = inputs[8]
            x_ground = inputs[9]
            y_ground = inputs[10]
            ground_mask = inputs[11]
            share_x_ground_mask = inputs[12]
            share_y_ground_mask = inputs[13]
            x_ground_mask = inputs[14]
            y_ground_mask = inputs[15]
            time = inputs[16]
            x_time = inputs[17]
            y_time = inputs[18]
            x_time_window = inputs[19]
            y_time_window = inputs[20]
            len_x = inputs[21]
            len_y = inputs[22]
            x_ground_mask_share = inputs[23]
            y_ground_mask_share = inputs[24]
            a_position = inputs[25]
            x_interval_time = inputs[26]
            y_interval_time = inputs[27]
            interval_time = inputs[28]
            raw_interval_time = inputs[29]
        return seq, x_seq, y_seq, position, x_position, y_position, ground, share_x_ground, share_y_ground, x_ground, y_ground, ground_mask, share_x_ground_mask, share_y_ground_mask, x_ground_mask, y_ground_mask, time, x_time, y_time, x_time_window, y_time_window, len_x, len_y, x_ground_mask_share, y_ground_mask_share, a_position, x_interval_time, y_interval_time, interval_time, raw_interval_time

    def train(self, global_step, train_batch):
        self.model.eval()
        self.Diffusion.eval()
        x_aug_seq = []
        x_aug_time = []
        y_aug_seq = []
        y_aug_time = []
        padding_num = self.opt["source_item_num"] + self.opt["target_item_num"]
        source_num = self.opt["source_item_num"]
        for batch in train_batch:
            seq, x_seq, y_seq, position, x_position, y_position, ground, share_x_ground, share_y_ground, x_ground, y_ground, ground_mask, share_x_ground_mask, share_y_ground_mask, x_ground_mask, y_ground_mask, time, x_time, y_time, x_time_window, y_time_window, len_x, len_y, x_ground_mask_share, y_ground_mask_share, a_position, x_interval_time, y_interval_time, interval_time, raw_interval_time = self.unpack_batch(
                batch)
            seqs_fea, x_seqs_fea, y_seqs_fea, item_emb_X, item_emb_Y = self.model(seq, x_seq, y_seq, position, x_position, y_position, time,
                                                          x_time, y_time, x_interval_time, y_interval_time, interval_time)

            x_mask = x_ground_mask / x_ground_mask.sum(dim=1, keepdim=True)
            r_x_fea = (x_seqs_fea * x_mask.unsqueeze(-1)).sum(dim=1)
            y_mask = y_ground_mask / y_ground_mask.sum(dim=1, keepdim=True)
            r_y_fea = (y_seqs_fea * y_mask.unsqueeze(-1)).sum(dim=1)

            x_mask = x_ground_mask_share / x_ground_mask_share.sum(dim=1, keepdim=True)
            real_x_fea = (seqs_fea * x_mask.unsqueeze(-1)).sum(dim=1)
            y_mask = y_ground_mask_share / y_ground_mask_share.sum(dim=1, keepdim=True)
            real_y_fea = (seqs_fea * y_mask.unsqueeze(-1)).sum(dim=1)

            x_avg_fea = r_x_fea + real_x_fea
            y_avg_fea = r_y_fea + real_y_fea

            aug_item, aug_time = generate_aug_item(self.opt, y_time, self.Diffusion, x_seq, x_time,
                                                                   len_x, x_avg_fea, item_emb_X, flag=0)

            x_aug_seq.append(aug_item)
            x_aug_time.append(aug_time)
            y_seq = torch.where(y_seq == padding_num, y_seq, y_seq - source_num).to(self.device)
            aug_item, aug_time = generate_aug_item(self.opt, x_time, self.Diffusion, y_seq, y_time,
                                                                   len_y, y_avg_fea, item_emb_Y, flag=1)
            aug_item = torch.where(aug_item == padding_num, aug_item, aug_item + source_num).to(self.device)
            y_aug_seq.append(aug_item)
            y_aug_time.append(aug_time)

        self.model.train()
        train_loss = 0
        for batch_idx, batch in enumerate(train_batch):
            global_step += 1
            current_x_aug = x_aug_seq[batch_idx]
            current_y_aug = y_aug_seq[batch_idx]
            current_x_time = x_aug_time[batch_idx]
            current_y_time = y_aug_time[batch_idx]

            loss = self.train_batch(batch, x_aug=current_x_aug, y_aug=current_y_aug, x_aug_time=current_x_time,
                                    y_aug_time=current_y_time)
            train_loss += loss

        self.Diffusion.train()
        dif_loss = 0
        for batch in train_batch:
            seq, x_seq, y_seq, position, x_position, y_position, ground, share_x_ground, share_y_ground, x_ground, y_ground, ground_mask, share_x_ground_mask, share_y_ground_mask, x_ground_mask, y_ground_mask, time, x_time, y_time, x_time_window, y_time_window, len_x, len_y, x_ground_mask_share, y_ground_mask_share, a_position, x_interval_time, y_interval_time, interval_time, raw_interval_time = self.unpack_batch(
                batch)
            seqs_fea, x_seqs_fea, y_seqs_fea, item_emb_X, item_emb_Y = self.model(seq, x_seq, y_seq, position, x_position, y_position, time,
                                                          x_time, y_time, x_interval_time, y_interval_time, interval_time)

            x_mask = x_ground_mask / x_ground_mask.sum(dim=1, keepdim=True)
            r_x_fea = (x_seqs_fea * x_mask.unsqueeze(-1)).sum(dim=1)
            y_mask = y_ground_mask / y_ground_mask.sum(dim=1, keepdim=True)
            r_y_fea = (y_seqs_fea * y_mask.unsqueeze(-1)).sum(dim=1)

            x_mask = x_ground_mask_share / x_ground_mask_share.sum(dim=1, keepdim=True)
            real_x_fea = (seqs_fea * x_mask.unsqueeze(-1)).sum(dim=1)
            y_mask = y_ground_mask_share / y_ground_mask_share.sum(dim=1, keepdim=True)
            real_y_fea = (seqs_fea * y_mask.unsqueeze(-1)).sum(dim=1)

            x_avg_fea = r_x_fea + real_x_fea
            y_avg_fea = r_y_fea + real_y_fea

            bpr_loss = self.Diffusion.calculate_loss(x_time, x_avg_fea, y_time, y_avg_fea)

            dif_loss += bpr_loss

            bpr_loss.backward()
            self.optimizer2.step()
            self.optimizer2.zero_grad()

        return global_step, train_loss, dif_loss

    def train_batch(self, batch, x_aug=None, y_aug=None, x_aug_time=None, y_aug_time=None):
        seq, x_seq, y_seq, position, x_position, y_position, ground, share_x_ground, share_y_ground, x_ground, y_ground, ground_mask, share_x_ground_mask, share_y_ground_mask, x_ground_mask, y_ground_mask, time, x_time, y_time, x_time_window, y_time_window, len_x, len_y, x_ground_mask_share, y_ground_mask_share, a_position, x_interval_time, y_interval_time, interval_time, raw_interval_time = self.unpack_batch(
            batch)
        seqs_fea, x_seqs_fea, y_seqs_fea, item_emb_X, item_emb_Y = self.model(seq, x_seq, y_seq, position, x_position, y_position, time, x_time,
                                                      y_time, x_interval_time, y_interval_time, interval_time)

        x_flase_fea, y_false_fea = self.model.false_forward(x_aug, y_aug, a_position, a_position, x_aug_time, y_aug_time, raw_interval_time, raw_interval_time)

        x_mask = x_ground_mask / x_ground_mask.sum(dim=1, keepdim=True)
        r_x_fea = (x_seqs_fea * x_mask.unsqueeze(-1)).sum(dim=1)
        y_mask = y_ground_mask / y_ground_mask.sum(dim=1, keepdim=True)
        r_y_fea = (y_seqs_fea * y_mask.unsqueeze(-1)).sum(dim=1)

        aug_mask = (a_position > 0).float()
        mask = aug_mask / aug_mask.sum(dim=1, keepdim=True)
        x_false_fea = (x_flase_fea * mask.unsqueeze(-1)).sum(dim=1)
        y_false_fea = (y_false_fea * mask.unsqueeze(-1)).sum(dim=1)

        def contrastive_loss(anchor, positive, temperature=0.1):
            batch_size = anchor.size(0)

            anchor_norm = F.normalize(anchor, dim=1)
            positive_norm = F.normalize(positive, dim=1)
            features = torch.cat([anchor_norm, positive_norm], dim=0)

            sim_matrix = torch.mm(features, features.T) / temperature

            labels = torch.arange(batch_size, device=anchor.device)
            pos_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
            pos_mask[labels, labels + batch_size] = 1
            pos_mask[labels + batch_size, labels] = 1

            neg_mask = torch.ones_like(sim_matrix, dtype=torch.bool)
            neg_mask.fill_diagonal_(0)
            neg_mask[pos_mask] = 0

            pos_sim = sim_matrix[pos_mask]
            neg_sim = sim_matrix[neg_mask]

            neg_sim = neg_sim.view(2 * batch_size, -1)
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
            labels = torch.zeros(2 * batch_size, dtype=torch.long, device=anchor.device)
            loss = F.cross_entropy(logits, labels)

            return loss

        cl_loss_x = contrastive_loss(r_x_fea, x_false_fea, 0.1)
        cl_loss_y = contrastive_loss(r_y_fea, y_false_fea, 0.1)

        used = 10
        share_x_ground = share_x_ground[:, -used:]
        share_x_ground_mask = share_x_ground_mask[:, -used:]
        share_y_ground = share_y_ground[:, -used:]
        share_y_ground_mask = share_y_ground_mask[:, -used:]
        x_ground = x_ground[:, -used:]
        x_ground_mask = x_ground_mask[:, -used:]
        y_ground = y_ground[:, -used:]
        y_ground_mask = y_ground_mask[:, -used:]

        share_x_result = self.model.lin_X(seqs_fea[:, -used:])
        share_y_result = self.model.lin_Y(seqs_fea[:, -used:])
        share_pad_result = self.model.lin_PAD(seqs_fea[:, -used:])
        share_trans_x_result = torch.cat((share_x_result, share_pad_result), dim=-1)
        share_trans_y_result = torch.cat((share_y_result, share_pad_result), dim=-1)

        x_tmp = self.model.weight_x(x_time_window)
        mix_fea = x_tmp.view(-1, 1, 1) * x_seqs_fea[:, -used:] + (1 - x_tmp.view(-1, 1, 1)) * seqs_fea[:, -used:]
        specific_x_result = self.model.lin_X(mix_fea)
        specific_x_pad_result = self.model.lin_PAD(x_seqs_fea[:, -used:])
        specific_x_result = torch.cat((specific_x_result, specific_x_pad_result), dim=-1)

        y_tmp = self.model.weight_y(y_time_window)
        mix_fea = y_tmp.view(-1, 1, 1) * y_seqs_fea[:, -used:] + (1 - y_tmp.view(-1, 1, 1)) * seqs_fea[:, -used:]
        specific_y_result = self.model.lin_Y(mix_fea)
        specific_y_pad_result = self.model.lin_PAD(y_seqs_fea[:, -used:])
        specific_y_result = torch.cat((specific_y_result, specific_y_pad_result), dim=-1)

        x_share_loss = self.CS_criterion(
            share_trans_x_result.reshape(-1, self.opt["source_item_num"] + 1),
            share_x_ground.reshape(-1))
        y_share_loss = self.CS_criterion(
            share_trans_y_result.reshape(-1, self.opt["target_item_num"] + 1),
            share_y_ground.reshape(-1))
        x_loss = self.CS_criterion(
            specific_x_result.reshape(-1, self.opt["source_item_num"] + 1),
            x_ground.reshape(-1))
        y_loss = self.CS_criterion(
            specific_y_result.reshape(-1, self.opt["target_item_num"] + 1),
            y_ground.reshape(-1))

        x_window = x_time_window.mean(dim=2, keepdim=False)
        y_window = y_time_window.mean(dim=2, keepdim=False)
        x_tmp = x_tmp.squeeze(1)
        y_tmp = y_tmp.squeeze(1)

        x_neg = torch.randint(0, self.opt['batch_size'], (self.opt['batch_size'],))
        y_neg = torch.randint(0, self.opt['batch_size'], (self.opt['batch_size'],))

        x_diff = x_window - x_window[x_neg]
        y_diff = y_window - y_window[y_neg]
        x_weight = x_tmp - x_tmp[x_neg]
        y_weight = y_tmp - y_tmp[y_neg]
        x_window_loss = torch.mean(torch.relu(x_weight * torch.sign(-x_diff)))
        y_window_loss = torch.mean(torch.relu(y_weight * torch.sign(-y_diff)))

        x_share_loss = (x_share_loss * (share_x_ground_mask.reshape(-1))).mean()
        y_share_loss = (y_share_loss * (share_y_ground_mask.reshape(-1))).mean()
        x_loss = (x_loss * (x_ground_mask.reshape(-1))).mean()
        y_loss = (y_loss * (y_ground_mask.reshape(-1))).mean()

        loss = x_share_loss + y_share_loss + x_loss + y_loss + 0.1 * (
                cl_loss_x + cl_loss_y) + self.opt["lambda"] * (x_window_loss + y_window_loss)

        self.mi_loss += 0.1 * (cl_loss_x.item() + cl_loss_y.item())
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def test_batch(self, batch):
        seq, x_seq, y_seq, position, x_position, y_position, X_last, Y_last, x_ground, y_ground, negative_sample_x, negative_sample_y, time, x_time, y_time, x_time_window, y_time_window, len_x, len_y, x_interval_time, y_interval_time, interval_time = self.unpack_batch_predict(
            batch)
        seqs_fea, x_seqs_fea, y_seqs_fea, item_emb_X, item_emb_Y = self.model(seq, x_seq, y_seq, position, x_position, y_position, time, x_time,
                                                      y_time, x_interval_time, y_interval_time, interval_time)

        X_pred = []
        Y_pred = []
        x_weight = self.model.weight_x(x_time_window)
        y_weight = self.model.weight_y(y_time_window)
        for id, fea in enumerate(seqs_fea):
            share_fea = seqs_fea[id, -1]
            specific_fea = x_seqs_fea[id, X_last[id]]
            final_fea = specific_fea * x_weight[id].item() + share_fea * (1 - x_weight[id].item())
            X_score = self.model.lin_X(final_fea).squeeze(0)
            cur = X_score[x_ground[id]]
            score_larger = (X_score[negative_sample_x[id]] > (cur + 0.00001)).data.cpu().numpy()
            true_item_rank = np.sum(score_larger) + 1
            X_pred.append(true_item_rank)

        for id, fea in enumerate(seqs_fea):
            share_fea = seqs_fea[id, -1]
            specific_fea = y_seqs_fea[id, Y_last[id]]
            final_fea = specific_fea * y_weight[id].item() + share_fea * (1 - y_weight[id].item())
            Y_score = self.model.lin_Y(final_fea).squeeze(0)
            cur = Y_score[y_ground[id]]
            score_larger = (Y_score[negative_sample_y[id]] > (cur + 0.00001)).data.cpu().numpy()
            true_item_rank = np.sum(score_larger) + 1
            Y_pred.append(true_item_rank)

        return X_pred, Y_pred
