import torch
from torch import nn
from torch import nn, Tensor


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + "." + name1 if name != "" else name1))
    return res


def init_quantization_scale(x: Tensor, n_bits, symmetric: bool, channel_wise: bool, scale_method: str = "mse", signed=True):
    # parallel batch
    n_batch = x.shape[0] if channel_wise else 1
    x_flat = x.reshape(n_batch, -1).detach()
    best_score = torch.full([n_batch], 1e10, device=x.device)

    # Four cases need to be considered: {signed, unsigned} x {symmetric, asymmetric}
    if symmetric:
        max_value = x_flat.abs().max(dim=1).values
        x_max = max_value
        x_min = -max_value if signed else torch.zeros_like(x_max)
    else:
        x_max = x_flat.max(dim=1).values
        x_min = x_flat.min(dim=1).values if signed else torch.max(x_flat.min(dim=1).values, torch.tensor([0.0]))
    delta = torch.zeros_like(best_score)
    zero_point = torch.zeros_like(best_score)

    # Finding scales in parallel
    for clip_ratio in torch.arange(1.0, 0.0, -0.01):
        new_max, new_min = x_max * clip_ratio, x_min * clip_ratio

        new_delta = (new_max - new_min) / (2**n_bits - 1)
        new_min = new_min if new_min.dtype != torch.float16 else new_min.to(torch.float32)
        new_delta = new_delta if new_delta.dtype != torch.float16 else new_delta.to(torch.float32)

        if symmetric:
            new_zeropoint = torch.ceil(-new_min / new_delta)
        else:
            new_zeropoint = torch.round(-new_min / new_delta)
        x_q = uniform_quantize(x_flat, new_delta.unsqueeze(1), new_zeropoint.unsqueeze(1), n_bits)

        if scale_method == "max":  # min-max clipping
            target_dim = [-1, *[1] * (len(x.shape) - 1)]
            return new_delta.view(target_dim).to(torch.float16), new_zeropoint.view(target_dim).to(torch.float16)
        elif scale_method == "mse":
            score = (x_flat - x_q).abs().pow(2.4).mean(dim=1)
        elif scale_method == "l1":
            score = (x_flat - x_q).abs().mean(dim=1)
        else:
            raise ValueError(f"Scale method {scale_method} is not exist!")
        delta = torch.where(score < best_score, new_delta, delta)
        zero_point = torch.where(score < best_score, new_zeropoint, zero_point)
        best_score = torch.minimum(score, best_score)
    if torch.any(delta < 1e-10):
        log.warning(f"Quantization range close to zero: [{delta}]")
    target_dim = [-1, *[1] * (len(x.shape) - 1)]
    return delta.view(target_dim), zero_point.view(target_dim)


def find_qkv_params(use_hessian, weight, n_bits, H):
    dev = weight.device
    maxq = torch.tensor(2**n_bits - 1).to(dev)
    shape = weight.shape
    weight = weight.flatten(1)  # perchannel

    tmp = torch.zeros(weight.shape[0], device=dev)
    xmin = torch.minimum(weight.min(1)[0], tmp)
    xmax = torch.maximum(weight.max(1)[0], tmp)

    # asymmetric
    tmp = (xmin == 0) & (xmax == 0)
    xmin[tmp] = -1
    xmax[tmp] = +1

    if maxq < 0:
        scale = xmax
        zero = xmin
    else:
        scale = (xmax - xmin) / maxq  # Min-Max
        zero = torch.round(-xmin / scale)

    shape = [-1] + [1] * (len(shape) - 1)
    scale = scale.reshape(shape)
    zero = zero.reshape(shape)
    gamma = torch.ones((1, weight.shape[1]), device=dev)
    scale, zeta, zero, diff, alternating_iter = find_zfold(use_hessian, weight, scale, zero, gamma, n_bits, H)  # EDITED 0814
    return scale, zeta, zero, maxq, diff, alternating_iter


def find_zfold(use_hessian, weight, delta, zero_point, zfold, n_bits, H):
    eps = 1e-10
    s_g = delta * zfold
    x_q = torch.clamp(torch.round(weight / (s_g + eps)) + zero_point, 0, 2**n_bits - 1)
    x_deq = (x_q - zero_point) * (s_g + eps)
    delta_W = x_deq - weight
    before_recon_loss = ((delta_W @ H) * delta_W).sum(dim=1).mean()

    first_recon_loss = before_recon_loss
    best_delta, best_zfold, best_zero = delta, zfold, zero_point
    final_iter = 0
    n_iters = 30
    if n_bits > 1:
        for iters in range(n_iters):
            w_q = uniform_quantize(weight / (zfold + eps), delta, zero_point, n_bits)
            zfold = mmse(w_q.transpose(0, 1), weight.transpose(0, 1)).view(zfold.shape)
            zfold = torch.where(zfold == 0.0, torch.ones(1).cuda(), zfold)
            if use_hessian:
                delta, zero_point = init_quantization_scale_H(
                    weight / (zfold + eps),
                    (zfold + eps),
                    n_bits=n_bits,
                    symmetric=False,
                    channel_wise=True,
                    scale_method="mse",
                    signed=True,
                    H=H,
                )
            else:
                delta, zero_point = init_quantization_scale(
                    weight / (zfold + eps), n_bits=n_bits, symmetric=False, channel_wise=True, scale_method="max", signed=True
                )

            s_g = delta * zfold
            x_q = torch.clamp(torch.round(weight / (s_g + eps)) + zero_point, 0, 2**n_bits - 1)
            x_deq = (x_q - zero_point) * (s_g + eps)
            delta_W = x_deq - weight
            after_recon_loss = ((delta_W @ H) * delta_W).sum(dim=1).mean()

            if before_recon_loss >= after_recon_loss:  # early stopping
                best_delta, best_zfold, best_zero = delta, zfold, zero_point
                before_recon_loss = after_recon_loss
                final_iter = iters + 1
            else:
                break
    diff = first_recon_loss - before_recon_loss
    return best_delta, best_zfold, best_zero, diff, final_iter


def uniform_quantize(x, delta, zero_point, n_bits):
    with torch.no_grad():
        x_int = torch.round(x / delta)
        x_q = torch.clamp(x_int + zero_point, 0, 2**n_bits - 1)
        x_deq = (x_q - zero_point) * delta
    return x_deq


def uniform_quantize_zeta(x, zeta, delta, zero_point, n_bits):
    with torch.no_grad():
        eps = 1e-10
        s_g = delta * zeta
        x_q = torch.clamp(torch.round(x / (s_g + eps)) + zero_point, 0, 2**n_bits - 1)
        x_deq = (x_q - zero_point) * (s_g + eps)
    return x_deq


def mmse(w_q, w):  # least squares := (w_qTw_q)^-1 (w_qTw) || (w_qTw)/(w_qTw_q)
    w_q = w_q.to(w.dtype)
    p = torch.bmm(w_q.unsqueeze(1), w.unsqueeze(2))
    q = torch.bmm(w_q.unsqueeze(1), w_q.unsqueeze(2))
    q = 1e-10 * torch.ones(q.shape).cuda() + q
    return p / q


def init_quantization_scale_H(x, zeta, n_bits, symmetric: bool, channel_wise: bool, scale_method: str = "mse", signed=True, H=None):
    n_batch = x.shape[0] if channel_wise else 1
    x_flat = x.reshape(n_batch, -1).detach()

    best_score = torch.full([n_batch], 1e10, device=x.device)
    if symmetric:
        max_value = x_flat.abs().max(dim=1).values
        x_max = max_value
        x_min = -max_value if signed else torch.zeros_like(x_max)
    else:
        x_max = x_flat.max(dim=1).values
        x_min = x_flat.min(dim=1).values if signed else torch.max(x_flat.min(dim=1).values, torch.tensor([0.0]))

    delta = torch.zeros_like(best_score)
    zero_point = torch.zeros_like(best_score)

    for clip_ratio in torch.arange(1.0, 0.0, -0.01):
        new_max, new_min = x_max * clip_ratio, x_min * clip_ratio

        new_delta = (new_max - new_min) / (2**n_bits - 1)
        for round in ("floor", "ceil"):
            if round == "floor":
                new_zeropoint = (-new_min / new_delta).floor()
            elif round == "ceil":
                new_zeropoint = (-new_min / new_delta).ceil()
            x_q = uniform_quantize(x_flat, new_delta.unsqueeze(1), new_zeropoint.unsqueeze(1), n_bits)

            if scale_method == "max":  # min-max clipping
                return new_delta, new_zeropoint
            elif scale_method == "mse":
                delta_W = (x_flat - x_q) * zeta
                if H is None:
                    score = (delta_W * delta_W).sum(dim=1)
                else:
                    # equivalent with torch.diag(delta_W @ H @ delta_W.T)
                    score = ((delta_W @ H) * delta_W).sum(dim=1)
            elif scale_method == "l1":
                score = (x_flat - x_q).abs().mean(dim=1)
            else:
                raise ValueError(f"Scale method {scale_method} is not exist!")

            delta = torch.where(score < best_score, new_delta, delta)
            zero_point = torch.where(score < best_score, new_zeropoint, zero_point)
            best_score = torch.minimum(score, best_score)
    target_dim = [-1, *[1] * (len(x.shape) - 1)]
    return delta.view(target_dim), zero_point.view(target_dim)


def find_only_scale(use_hessian, weight, delta, zero_point, zfold, n_bits, H):
    eps = 1e-10
    s_g = delta * zfold
    x_q = torch.clamp(torch.round(weight / (s_g + eps)) + zero_point, 0, 2**n_bits - 1)
    x_deq = (x_q - zero_point) * (s_g + eps)
    delta_W = x_deq - weight
    # before_recon_loss = (torch.diag(H)*(x_deq - weight)).abs().pow(2.4).sum(1).mean()
    before_recon_loss = ((delta_W @ H) * delta_W).sum(dim=1).mean()

    if use_hessian:
        delta, zero_point = init_quantization_scale_H(
            weight / (zfold + eps), (zfold + eps), n_bits=n_bits, symmetric=False, channel_wise=True, scale_method="mse", signed=True, H=H
        )
    else:
        delta, zero_point = init_quantization_scale(
            weight / (zfold + eps), n_bits=n_bits, symmetric=False, channel_wise=True, scale_method="max", signed=True
        )
    s_g = delta * zfold
    x_q = torch.clamp(torch.round(weight / s_g) + zero_point, 0, 2**n_bits - 1)
    x_deq = (x_q - zero_point) * (s_g + eps)
    delta_W = x_deq - weight
    # after_recon_loss = (torch.diag(H)*(x_deq - weight)).abs().pow(2.4).sum(1).mean()
    after_recon_loss = ((delta_W @ H) * delta_W).sum(dim=1).mean()
    diff = before_recon_loss - after_recon_loss
    return delta, zfold, zero_point, diff


def quantize_zfold(x, scale, zero, zeta, maxq):
    s_g = scale * zeta
    q = torch.clamp(torch.round(x / s_g) + zero, 0, maxq)
    return (s_g * (q - zero)).squeeze(dim=0)
