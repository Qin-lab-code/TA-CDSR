import enum
import math
import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from model.layers import AutoEncoderMixin
from model.layers import MLPLayers
from model.init import xavier_normal_initialization
from torch import Tensor
import typing
import scipy.sparse as sp
Enum = None


class InputType():

    POINTWISE = 1
    PAIRWISE = 2
    LISTWISE = 3


class ModelMeanType(enum.Enum):
    START_X = enum.auto()
    EPSILON = enum.auto()


class GraphVAEEncoder(nn.Module):
    def __init__(self, hidden_size):
        super(GraphVAEEncoder, self).__init__()

        self.fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)

        self.fc_mu = nn.Linear(hidden_size, hidden_size)
        self.fc_logvar = nn.Linear(hidden_size, hidden_size)

    def forward(self, itemEmb):
        h = F.relu(self.fc1(itemEmb))
        h = F.relu(self.fc2(h))

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps

        return z, mu, logvar


class DNN(nn.Module):

    def __init__(
            self,
            dims: typing.List,
            emb_size: int,
            hid_temp: int,
            time_type="cat",
            act_func="tanh",
            norm=False,
            dropout=0.4,
    ):
        super(DNN, self).__init__()
        self.dims = dims
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm
        self.hid_temp = hid_temp

        self.temp_enc = TimeEncode(self.hid_temp)

        if self.time_type == "cat":
            self.dims[0] += self.time_emb_dim
            self.dims[0] += self.hid_temp
        else:
            raise ValueError(
                "Unimplemented timestep embedding type %s" % self.time_type
            )

        self.mlp_layers = MLPLayers(
            layers=self.dims, dropout=0, activation=act_func, last_activation=False
        )
        self.drop = nn.Dropout(dropout)

        self.apply(xavier_normal_initialization)

    def forward(self, x, steps, id_time):
        embeds_time = self.temp_enc(id_time)
        steps_emb = timestep_embedding(steps, self.time_emb_dim).to(x.device)
        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)
        h = torch.cat([x, steps_emb, embeds_time], dim=-1)
        h = self.mlp_layers(h)
        return h

class TimeEncode(nn.Module):
    def __init__(self, time_dim, factor=5):
        super().__init__()
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())


    def forward(self, ts):
        map_ts = ts.unsqueeze(-1) * self.basis_freq.view(1, 1, -1)
        map_ts += self.phase.view(1, 1, -1)
        harmonic = torch.cos(map_ts)
        return harmonic.squeeze(0)


class TimeDiff(nn.Module, AutoEncoderMixin):

    input_type = InputType.LISTWISE

    def __init__(self, opt, device, mean_type,
                 noise_schedule, noise_scale, noise_max, noise_min, steps, beta_fixed, embedding_size, emb_size, hid_temp, norm, reweight,
                 sampling_steps, sampling_noise, mlp_act_func, history_num_per_term, dims_dnn):
        super(TimeDiff, self).__init__()
        self.opt = opt
        self.device = device
        if mean_type == "x0":
            self.mean_type = ModelMeanType.START_X
        elif mean_type == "eps":
            self.mean_type = ModelMeanType.EPSILON
        else:
            raise ValueError("Unimplemented mean type %s" % mean_type)

        self.noise_schedule = noise_schedule
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps
        self.beta_fixed = beta_fixed
        self.t_size = embedding_size
        self.emb_size = emb_size
        self.hid_temp = hid_temp
        self.norm = norm
        self.reweight = reweight
        if self.noise_scale == 0.0:
            self.reweight = False
        self.sampling_noise = sampling_noise
        self.sampling_steps = sampling_steps
        self.mlp_act_func = mlp_act_func
        assert self.sampling_steps <= self.steps, "Too much steps in inference."

        self.history_num_per_term = history_num_per_term
        self.Lt_history = torch.zeros(
            self.steps, self.history_num_per_term, dtype=torch.float64
        ).to(self.device)
        self.Lt_count = torch.zeros(self.steps, dtype=int).to(self.device)

        dims_x = [emb_size] + dims_dnn + [emb_size]
        dims_y = [emb_size] + dims_dnn + [emb_size]

        self.mlp_x = DNN(dims=dims_x, emb_size=self.t_size, hid_temp=self.hid_temp, time_type="cat", norm=self.norm,
                       act_func=self.mlp_act_func,).to(self.device)
        self.mlp_y = DNN(dims=dims_y, emb_size=self.t_size, hid_temp=self.hid_temp, time_type="cat", norm=self.norm,
                       act_func=self.mlp_act_func, ).to(self.device)

        self.vae_encoder_x = GraphVAEEncoder(self.emb_size)
        self.vae_encoder_y = GraphVAEEncoder(self.emb_size)

        if self.noise_scale != 0.0:
            self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).to(
                self.device
            )
            if self.beta_fixed:
                self.betas[0] = (
                    0.00001
                )
            assert len(self.betas.shape) == 1, "betas must be 1-D"
            assert (
                    len(self.betas) == self.steps
            ), "num of betas must equal to diffusion steps"
            assert (self.betas > 0).all() and (
                    self.betas <= 1
            ).all(), "betas out of range"

            self.calculate_for_diffusion()


    def get_betas(self):

        if self.noise_schedule == "linear" or self.noise_schedule == "linear-var":
            start = self.noise_scale * self.noise_min
            end = self.noise_scale * self.noise_max
            if self.noise_schedule == "linear":
                return np.linspace(start, end, self.steps, dtype=np.float64)
            else:
                return betas_from_linear_variance(
                    self.steps, np.linspace(start, end, self.steps, dtype=np.float64)
                )
        elif self.noise_schedule == "cosine":
            return betas_for_alpha_bar(
                self.steps, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            )
        elif self.noise_schedule == "binomial":
            ts = np.arange(self.steps)
            betas = [1 / (self.steps - t + 1) for t in ts]
            return betas
        else:
            raise NotImplementedError(f"unknown beta schedule: {self.noise_schedule}!")

    def calculate_for_diffusion(self):
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0).to(self.device)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]).to(self.device), self.alphas_cumprod[:-1]]
        ).to(self.device)
        self.alphas_cumprod_next = torch.cat(
            [self.alphas_cumprod[1:], torch.tensor([0.0]).to(self.device)]
        ).to(self.device)
        assert self.alphas_cumprod_prev.shape == (self.steps,)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
                self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        self.posterior_log_variance_clipped = torch.log(
            torch.cat(
                [self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]
            )
        )
        self.posterior_mean_coef1 = (
                self.betas
                * torch.sqrt(self.alphas_cumprod_prev)
                / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * torch.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
        )

    def p_sample(self, x_start, id_time, flag):

        steps = self.sampling_steps
        if steps == 0:
            x_t = x_start
        else:
            t = torch.tensor([steps - 1] * x_start.shape[0]).to(x_start.device)
            x_t = self.q_sample(x_start, t)

        indices = list(range(self.steps))[::-1]

        if self.noise_scale == 0.0:
            for i in indices:
                t = torch.tensor([i] * x_t.shape[0]).to(x_start.device)
                if flag == 0:
                    x_t = self.mlp_x(x_t, t, id_time)
                else:
                    x_t = self.mlp_y(x_t, t, id_time)
            return x_t

        for i in indices:
            t = torch.tensor([i] * x_t.shape[0]).to(x_start.device)
            out = self.p_mean_variance(x_t, t, id_time, flag)
            if self.sampling_noise:
                noise = torch.randn_like(x_t)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
                )
                x_t = (
                        out["mean"]
                        + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
                )
            else:
                x_t = out["mean"]
        return x_t

    def full_sort_predict(self, add_time, seqEmb, flag):
        if flag == 0:
            x_start, mu, logvar = self.vae_encoder_x(seqEmb)
        else:
            x_start, mu, logvar = self.vae_encoder_y(seqEmb)
        scores = self.p_sample(x_start, add_time, flag)
        return scores

    def calculate_loss(self, x_time, x_emb, y_time, y_emb):
        x_start, mu, logvar = self.vae_encoder_x(x_emb)
        y_start, mu, logvar = self.vae_encoder_y(y_emb)
        id_time_x = x_time[:, -1]
        id_time_y = y_time[:, -1]
        batch_size = x_start.size(0)
        ts_x = torch.randint(0, self.steps, (batch_size,)).long()
        ts_y = torch.randint(0, self.steps, (batch_size,)).long()
        if self.opt["cuda"]:
            ts_x = ts_x.cuda()
            ts_y = ts_y.cuda()
        noise_x = torch.randn_like(x_start)
        noise_y = torch.randn_like(y_start)
        if self.noise_scale != 0:
            x_t = self.q_sample(x_start, ts_x, noise_x)
            y_t = self.q_sample(y_start, ts_y, noise_y)
        else:
            x_t = x_start
            y_t = y_start

        model_output_x = self.mlp_x(x_t, ts_x, id_time_x)
        mse_x = mean_flat((x_start - model_output_x) ** 2)
        model_output_y = self.mlp_y(y_t, ts_y, id_time_y)
        mse_y = mean_flat((y_start - model_output_y) ** 2)

        weight_x = self.SNR(ts_x - 1) - self.SNR(ts_x)
        weight_x = torch.where((ts_x == 0), 1.0, weight_x)
        weight_y = self.SNR(ts_y - 1) - self.SNR(ts_y)
        weight_y = torch.where((ts_y == 0), 1.0, weight_y)

        diff_loss_x = (weight_x * mse_x).mean()
        diff_loss_y = (weight_y * mse_y).mean()

        l1_loss_x = torch.norm(model_output_x, p=1)
        l1_loss_y = torch.norm(model_output_y, p=1)
        alpha = 0.0001

        x_loss = l1_loss_x * alpha + diff_loss_x
        y_loss = l1_loss_y * alpha + diff_loss_y

        return x_loss + y_loss


    def sample_timesteps(
            self, batch_size, device, method="uniform", uniform_prob=0.001
    ):
        if method == "importance":
            if not (self.Lt_count == self.history_num_per_term).all():
                return self.sample_timesteps(batch_size, device, method="uniform")

            Lt_sqrt = torch.sqrt(torch.mean(self.Lt_history ** 2, axis=-1))
            pt_all = Lt_sqrt / torch.sum(Lt_sqrt)
            pt_all *= 1 - uniform_prob
            pt_all += uniform_prob / len(pt_all)

            assert pt_all.sum(-1) - 1.0 < 1e-5

            t = torch.multinomial(pt_all, num_samples=batch_size, replacement=True)
            pt = pt_all.gather(dim=0, index=t) * len(pt_all)

            return t, pt

        elif method == "uniform":
            t = torch.randint(0, self.steps, (batch_size,), device=device).long()
            pt = torch.ones_like(t).float()

            return t, pt

        else:
            raise ValueError

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
                self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
                * x_start
                + self._extract_into_tensor(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
                * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
                self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract_into_tensor(
            self.posterior_variance, t, x_t.shape
        )
        posterior_log_variance_clipped = self._extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, id_time, flag):

        B, C = x.shape[:2]
        assert t.shape == (B,)
        if flag == 0:
            model_output = self.mlp_x(x, t, id_time)
        else:
            model_output = self.mlp_y(x, t, id_time)

        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped

        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

        if self.mean_type == ModelMeanType.START_X:
            pred_xstart = model_output
        elif self.mean_type == ModelMeanType.EPSILON:
            pred_xstart = self._predict_xstart_from_eps(x, t, eps=model_output)
        else:
            raise NotImplementedError(self.mean_type)

        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        assert (
                model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
                * x_t
                - self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
                * eps
        )

    def SNR(self, t):

        self.alphas_cumprod = self.alphas_cumprod.to(t.device)
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):

        arr = arr.to(timesteps.device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)


def betas_from_linear_variance(steps, variance, max_beta=0.999):
    alpha_bar = 1 - variance
    betas = []
    betas.append(1 - alpha_bar[0])
    for i in range(1, steps):
        betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
    return np.array(betas)


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def normal_kl(mean1, logvar1, mean2, logvar2):

    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + torch.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def mean_flat(tensor):

    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def timestep_embedding(timesteps, dim, max_period=10000):

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(
        timesteps.device
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
