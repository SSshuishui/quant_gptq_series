import math
import time

import torch
import torch.nn as nn
import transformers

from .quant import *

DEBUG = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class AWRQ:

    def __init__(self, layer, method='awrq'):
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
        if DEBUG:
            self.dH = torch.zeros((self.columns, self.columns), device=self.dev)
        self.M = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.act_bits = layer.act_bits
        self.act_scales = torch.zeros(self.columns, device=self.dev, dtype=W.dtype)
        self.method = method
    
    def add_batch_act_scales(self, inp, out):
        inp_max = torch.max(inp.abs().reshape(-1,inp.shape[-1]), dim=0)[0]
        self.act_scales = torch.max(self.act_scales, inp_max)

    def add_batch(self, inp, out):
        # smooth: only for out_proj, fc2
        if self.layer.smooth_scales is not None and not self.layer.act_smoothed:
            inp.data /= self.layer.smooth_scales

        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, SmoothAndQuantLinear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)

        if self.method == 'gptq':
            self.H *= self.nsamples / (self.nsamples + tmp)
            if DEBUG:
                self.dH *= self.nsamples / (self.nsamples + tmp)
            self.nsamples += tmp
            inp = math.sqrt(1 / self.nsamples) * inp.float()
            self.H += inp.matmul(inp.t()) # must before quant
            if DEBUG:
                inp_quant = quantize_activation_per_tensor_asym(inp, self.act_bits)
                delta_inp = inp - inp_quant
                self.dH += delta_inp.matmul(delta_inp.t())
        elif self.method == 'awrq':
            self.H *= self.nsamples / (self.nsamples + tmp)
            if DEBUG:
                self.dH *= self.nsamples / (self.nsamples + tmp)
            self.M *= self.nsamples / (self.nsamples + tmp)
            self.nsamples += tmp
            inp = math.sqrt(1 / self.nsamples) * inp.float()
            inp_quant = quantize_activation_per_tensor_asym(inp, self.act_bits)
            self.H += inp_quant.matmul(inp_quant.t())
            delta_inp = inp - inp_quant
            self.M += delta_inp.matmul(inp_quant.t())
            if DEBUG:
                self.dH += delta_inp.matmul(delta_inp.t())

        if DEBUG:
            self.qinp1 = (inp_quant.clone().t()/math.sqrt(1 / self.nsamples)).to(self.layer.weight.dtype)

    def quant_weight(self, percdamp=.01, blocksize=1, groupsize=-1, method='awrq', actorder=False):

        if method not in ['gptq', 'rtn', 'smoothquant', 'awrq']:
            return
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
        M = self.M
        del self.M
        if DEBUG:
            dH = self.dH
            del self.dH
        if method in ['gptq', 'awrq']:
            dead = torch.diag(H) == 0
            H[dead, dead] = 1
            W[:, dead] = 0

            if actorder:
                perm = torch.argsort(torch.diag(H), descending=True)
                W = W[:, perm]
                H = H[perm][:, perm]
                M = M[perm][:, perm]

            Err1 = torch.zeros_like(W)
            Err2 = torch.zeros_like(W)
            Err3 = torch.zeros_like(W)
            Err  = torch.zeros_like(W)

            damp = percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(self.columns, device=self.dev)
            H[diag, diag] += damp

            U = torch.linalg.cholesky(H)
            U = torch.cholesky_inverse(U)
            U = torch.linalg.cholesky(U, upper=True)
            G = W.matmul(M)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        if DEBUG:
            Losses[:, 0] = ((self.layer.weight.float().matmul(dH))*self.layer.weight.float()).sum(dim=1) # const error 
            print('before layer error: error={:.4f}'.format(torch.sum((self.layer(self.qinp1) - self.out1) ** 2)))
            print('before quant error: error={:.4f}'.format(torch.sum(Losses)))
        if blocksize == -1:
            blocksize = self.columns//100 if self.columns//100 > 0 else 1
        if groupsize == -1 and method in ['rtn', 'smoothquant']:
            blocksize = self.columns

        for i in range(0, self.columns, blocksize):
            i2 = min(i+blocksize, self.columns)
            w = W[:, i:i2]

            if groupsize != -1:
                if i % groupsize == 0:
                    self.quantizer.find_params(W[:, i:(i + groupsize)], weight=True)

            q = quantize(w, self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq)
            Q[:, i:i2] = q

            if method in ['rtn', 'smoothquant']:
                continue
            # Gi, H1
            Err2[:, i:i2] = q - w # block quantization error
            H1 = U[i:i2,i:i2].t().matmul(U[i:i2,i:])
            if i == 0 and method == 'awrq': # AWRQ
                # Err
                Err1[:,i:] = G[:,i:].matmul(U.t()[i:,i:]).matmul(U[i:,i:])
                Err3[:,i:i2] = G[:,i:].matmul(H1.t())

                # correction
                Err[:,i:] = -Err1[:,i:] - ((Err2[:,i:i2] - Err3[:,i:i2]).matmul(torch.cholesky_inverse(U[i:i2,i:i2], upper=True))).matmul(H1)
                
                # Loss
                if DEBUG:
                    Losses[:, i] += ((Err[:,i:].matmul(H[i:,i:]))*Err[:,i:]).sum(dim=1) + 2*(G[:,i:]*Err[:,i:]).sum(dim=1)

                if DEBUG:
                    del dH 
                del G
                del Err1
                del Err3

            else:
                Err[:,i:] = - (Err2[:,i:i2].matmul(torch.cholesky_inverse(U[i:i2,i:i2], upper=True))).matmul(H1)
                # Loss
                if DEBUG:
                    Losses[:, i] += ((Err[:,i:].matmul(H[i:,i:]))*Err[:,i:]).sum(dim=1) 

            W[:,i:] = W[:,i:] - Err[:,i:] # not update column i

            if DEBUG:
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print('layer error: i={}, error={:.4f}'.format(i, torch.sum((self.layer(self.qinp1) - self.out1) ** 2)))
                print('quant error: i={}, error={:.4f}'.format(i, torch.sum(Losses)))


        if actorder:
            invperm = torch.argsort(perm)
            Q = Q[:, invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.qinp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
