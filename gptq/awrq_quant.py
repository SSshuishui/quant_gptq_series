import numpy as np
import torch
import torch.nn as nn

def quantize_activation_per_tensor_asym(inps, bits=8):
    inps_res = inps.clone()
    maxq = torch.tensor(2 ** bits - 1)
    xmax = torch.max(inps)
    xmin = torch.min(inps)
    scale = (xmax - xmin) / maxq
    zero = torch.round(-xmin / scale)
    q = torch.clamp(torch.round(inps / scale) + zero, 0, maxq)
    inps_res = scale * (q - zero)
    return inps_res

def quantize_activation_per_token_asym(inps, bits=8):
    inps_res = torch.zeros_like(inps)
    maxq = torch.tensor(2 ** bits - 1)
    xmax = torch.max(inps, dim=1, keepdim=True)[0]
    xmin = torch.min(inps, dim=1, keepdim=True)[0]
    scale = (xmax - xmin) / maxq
    scale = torch.clamp(scale, min=EPS, max=1e4)
    zero = torch.round(-xmin / scale)
    inps_res = inps / scale
    q = torch.clamp(torch.round(inps_res) + zero, 0, maxq)
    inps_res = scale * (q - zero)
    return inps_res


def quantize(x, scale, zero, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

class AWRQQuantizer(nn.Module):

    def __init__(self, shape=1):
        super(AWRQQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self,
        bits, act_bits=8, perchannel=False, sym=True, 
        mse=False, norm=2.4, grid=100, maxshrink=.8,
        trits=False
    ):
        self.maxq = torch.tensor(2 ** bits - 1)
        self.act_bits = act_bits # act_bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink 
        if trits:
            self.maxq = torch.tensor(-1) 

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
          self.scale = xmax
          self.zero = xmin
        else:
          self.scale = (xmax - xmin) / self.maxq
          if self.sym:
              self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
          else:
              self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid 
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1)) 
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


# Assumes layer is perfectly divisible into 1024 * 1024 blocks
class Quant3Linear(nn.Module): 

    def __init__(self, infeatures, outfeatures, faster=False):
        super().__init__()
        self.register_buffer('zeros', torch.zeros((outfeatures, 1)))
        self.register_buffer('scales', torch.zeros((outfeatures, 1)))
        self.register_buffer('bias', torch.zeros(outfeatures))
        self.register_buffer(
            'qweight', torch.zeros((infeatures // 32 * 3, outfeatures), dtype=torch.int)
        )
        self.faster = faster

    def pack(self, linear, scales, zeros):
        self.zeros = zeros * scales
        self.scales = scales.clone()
        self.bias = linear.bias.clone()

        intweight = torch.round((linear.weight.data + self.zeros) / self.scales).to(torch.int)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)
        qweight = np.zeros(
            (intweight.shape[0] // 32 * 3, intweight.shape[1]), dtype=np.uint32
        )
        i = 0
        row = 0
        while row < qweight.shape[0]:
            for j in range(i, i + 10):
                qweight[row] |= intweight[j] << (3 * (j - i))
            i += 10
            qweight[row] |= intweight[i] << 30
            row += 1
            qweight[row] |= (intweight[i] >> 2) & 1
            i += 1
            for j in range(i, i + 10):
                qweight[row] |= intweight[j] << (3 * (j - i) + 1)
            i += 10
            qweight[row] |= intweight[i] << 31
            row += 1
            qweight[row] |= (intweight[i] >> 1) & 0x3
            i += 1
            for j in range(i, i + 10):
                qweight[row] |= intweight[j] << (3 * (j - i) + 2)
            i += 10
            row += 1

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight) 

    def forward(self, x):
        if x.shape[-1] == x.numel():
            outshape = list(x.shape)
            y = self.bias.clone()
            outshape[-1] = self.bias.numel()
            dtype = x.dtype
            if self.faster:
                x = x.half()
                quant_cuda.vecquant3matmul_faster(x, self.qweight, y, self.scales, self.zeros)
            else:
                x = x.float()
                quant_cuda.vecquant3matmul(x, self.qweight, y, self.scales, self.zeros)
            y = y.to(dtype)
            return y.reshape(outshape)
        raise ValueError('Only supports a single token currently.')

def make_quant3(module, names, name='', faster=False):
    if isinstance(module, Quant3Linear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            setattr(
                module, attr, Quant3Linear(tmp.in_features, tmp.out_features, faster=faster)
            )
    for name1, child in module.named_children():
        make_quant3(child, names, name + '.' + name1 if name != '' else name1, faster=faster)

class SmoothAndQuantLinear(nn.Module): 

    def __init__(self, in_features, out_features, bias=True, device=torch.device('cuda'), smooth=False, alpha=0.50, min=0.01, act_quant=False, act_bits=8, act_pertoken=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('weight', torch.randn(self.out_features, self.in_features, dtype=torch.float16, requires_grad=False, device=device))
        if bias:
            self.register_buffer('bias', torch.zeros((1, self.out_features), dtype=torch.float16, requires_grad=False, device=device))
        else:
            self.register_buffer('bias', None)
        self.dev = self.weight.device
        self.smooth = smooth
        self.act_smoothed = False
        self.alpha = alpha
        self.min = min
        self.act_quant = act_quant
        self.act_pertoken = act_pertoken
        self.act_bits = act_bits
        self.smooth_scales = None 
        self.act_scales = None 
        self.weight_scales = None

    @staticmethod
    def from_float(module, smooth=False, alpha=0.50, min=0.01, act_quant=False, act_bits=8, act_pertoken=False):
        assert isinstance(module, torch.nn.Linear)
        new_module = SmoothAndQuantLinear(module.in_features, module.out_features, module.bias is not None, device=module.weight.device, smooth=smooth, alpha=alpha, min=min, act_quant=act_quant, act_bits=act_bits)
        new_module.weight = module.weight
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def forward(self, x):
        # smooth
        if self.smooth:
            # smooth scales
            if self.act_scales is not None and self.smooth_scales is None: 
                # weight scales
                if self.weight_scales is None:
                    self.weight_scales = self.weight.abs().max(dim=0)[0]
                self.smooth_scales = (self.act_scales.pow(self.alpha) / self.weight_scales.pow(1-self.alpha)).clamp(min=self.min)
            # import ipdb; ipdb.set_trace()
            if self.smooth_scales is not None and not self.act_smoothed:
                x = x/self.smooth_scales # smooth

        # quant
        if self.act_quant:
            if self.act_pertoken:
                x = quantize_activation_per_token_asym(x, self.act_bits)
            else:
                x = quantize_activation_per_tensor_asym(x, self.act_bits)

        y = torch.functional.F.linear(x, self.weight, self.bias)
        return y

def find_layers(module, layers=[nn.Conv2d, nn.Linear, SmoothAndQuantLinear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def replace_linear_layer(model_name, module, smooth=False, alpha=0.50, min=0.01, act_quant=False, act_bits=8):
    # replace OPTAttention
    module.self_attn.q_proj = SmoothAndQuantLinear.from_float(module.self_attn.q_proj, smooth=smooth, alpha=alpha, min=min, act_quant=act_quant, act_bits=act_bits)
    module.self_attn.k_proj = SmoothAndQuantLinear.from_float(module.self_attn.k_proj, smooth=smooth, alpha=alpha, min=min, act_quant=act_quant, act_bits=act_bits)
    module.self_attn.v_proj = SmoothAndQuantLinear.from_float(module.self_attn.v_proj, smooth=smooth, alpha=alpha, min=min, act_quant=act_quant, act_bits=act_bits)

    if 'opt' in model_name:
        module.self_attn.out_proj = SmoothAndQuantLinear.from_float(module.self_attn.out_proj, smooth=smooth, alpha=alpha, min=min, act_quant=act_quant, act_bits=act_bits)
        module.fc1 = SmoothAndQuantLinear.from_float(module.fc1, smooth=smooth, alpha=alpha, min=min, act_quant=act_quant, act_bits=act_bits)
        module.fc2 = SmoothAndQuantLinear.from_float(module.fc2, smooth=smooth, alpha=alpha, min=min, act_quant=act_quant, act_bits=act_bits)
    elif 'llama' in model_name:
        module.self_attn.o_proj = SmoothAndQuantLinear.from_float(module.self_attn.o_proj, smooth=smooth, alpha=alpha, min=min, act_quant=act_quant, act_bits=act_bits)
        module.mlp.gate_proj = SmoothAndQuantLinear.from_float(module.mlp.gate_proj, smooth=smooth, alpha=alpha, min=min, act_quant=act_quant, act_bits=act_bits)
        module.mlp.up_proj = SmoothAndQuantLinear.from_float(module.mlp.up_proj, smooth=smooth, alpha=alpha, min=min, act_quant=act_quant, act_bits=act_bits)
        module.mlp.down_proj = SmoothAndQuantLinear.from_float(module.mlp.down_proj, smooth=smooth, alpha=alpha, min=min, act_quant=act_quant, act_bits=act_bits)
    else:
        print(f'Not supported model: {model_name}')

    return module