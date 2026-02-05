import torch, torch.nn as nn
import time as tm


def generate_aug_item(opt, time_slice, Diffusion, item_seq, tensor_times, seq_len, seqEmb, item_emb, flag):
    item_list, time_list = zip(*[(
        item_seq[i, -seq_len[i].item():].tolist(),
        tensor_times[i, -seq_len[i].item():].tolist())
        for i in range(len(item_seq))])

    for idx in range(len(time_slice[0])):
        add_time_tensor = time_slice[:, idx]
        if opt['cuda']:
            add_time_tensor = add_time_tensor.cuda()
        is_zero = (torch.count_nonzero(add_time_tensor) == 0)
        if is_zero:
            continue
        else:
            x = Diffusion.full_sort_predict(add_time_tensor, seqEmb, flag)

            x = torch.matmul(x, item_emb.T)

            topk_indices = x.topk(k=1, dim=1).indices.squeeze(1)

            valid_mask = (add_time_tensor != 0)
            valid_indices = torch.where(valid_mask)[0]

            for idx in valid_indices:
                item_list[idx].append(topk_indices[idx].item())
                time_list[idx].append(add_time_tensor[idx].item())

    device = 'cuda' if opt['cuda'] else 'cpu'
    max_len = opt["maxlen"]
    pad_value = opt["source_item_num"] + opt["target_item_num"]
    batch_size = len(item_list)

    padded_items = torch.full((batch_size, max_len), pad_value, dtype=torch.long, device=device)
    padded_times = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

    for i in range(batch_size):
        seq_len = len(item_list[i])
        start_idx = max_len - seq_len

        padded_items[i, start_idx:] = torch.as_tensor(
            item_list[i][-max_len:],
            dtype=torch.long,
            device=device
        )
        padded_times[i, start_idx:] = torch.as_tensor(
            time_list[i][-max_len:],
            dtype=torch.long,
            device=device
        )

    sorted_indices = padded_times.argsort(dim=1)
    sorted_items = torch.gather(padded_items, 1, sorted_indices).to(device)
    sorted_times = torch.gather(padded_times, 1, sorted_indices).to(device)

    return sorted_items, sorted_times
