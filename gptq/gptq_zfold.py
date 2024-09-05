import math
import time

import torch
import torch.nn as nn
import transformers
from .zfold import *
from .quant import *

DEBUG = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(self.layer.kernel_size, dilation=self.layer.dilation, padding=self.layer.padding, stride=self.layer.stride)
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        static_groups=False,
        actorder=False,
        name="",
        ith=0,
        use_hessian=False,
        use_zfold=False,
        share_zeta=False,
        other_zeta=None,
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            import copy

            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i : (i + groupsize)], weight=True)
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)
            self.quantizer.zeta = self.quantizer.zeta.squeeze(1)[:, perm]

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp

        ttick = time.time()  # additional spending times for Z-fold
        update = False
        alternating_iter = 0
        if use_zfold:
            if share_zeta:
                """"""
                self.scale, self.zeta, self.zero, diff = find_only_scale(
                    use_hessian,
                    W,
                    self.quantizer.scale.squeeze(0),
                    self.quantizer.zero.squeeze(0),
                    self.quantizer.zeta.squeeze(0),
                    self.quantizer.nbits,
                    H,
                )
                self.quantizer.scale = self.scale
                self.quantizer.zero = self.zero
                self.quantizer.zeta = self.zeta.view((1, -1))
            else:
                self.scale, self.zeta, self.zero, diff, alternating_iter = find_zfold(
                    use_hessian,
                    W,
                    self.quantizer.scale.squeeze(0),
                    self.quantizer.zero.squeeze(0),
                    self.quantizer.zeta.squeeze(0),
                    self.quantizer.nbits,
                    H,
                )
                self.quantizer.scale = self.scale
                self.quantizer.zero = self.zero
                self.quantizer.zeta = self.zeta.view((1, -1))
        else:
            self.scale, self.zeta, self.zero, diff = find_only_scale(
                use_hessian,
                W,
                self.quantizer.scale.squeeze(0),
                self.quantizer.zero.squeeze(0),
                self.quantizer.zeta.squeeze(0),
                self.quantizer.nbits,
                H,
            )
            self.quantizer.scale = self.scale
            self.quantizer.zero = self.zero
            self.quantizer.zeta = self.zeta.view((1, -1))

        print(f"|{ith}: {name}\t| {diff:.3f}\t| {(time.time() - ttick):.2f}\t| {alternating_iter}\t|")

        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        # OPTQ
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i) : (i1 + i + groupsize)], weight=True)
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                zfold_idx = i1 + i
                q = quantize_zfold(
                    w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.zeta[:, zfold_idx], self.quantizer.maxq
                ).flatten()
                Q1[:, i] = q

                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        if actorder:
            if len(Q.shape) == 3:
                Q = Q.squeeze(0)
            Q = Q[:, invperm]
            self.quantizer.zeta = self.quantizer.zeta[:, invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
