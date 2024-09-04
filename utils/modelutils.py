import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaRMSNorm

DEV = torch.device('cuda:0')

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        obj = obj.to(device)
        return obj
    elif isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            new_obj[k] = to_device(v, device)
        return new_obj
    elif isinstance(obj, (list, tuple)):
        new_obj = []
        for v in obj:
            new_obj.append(to_device(v, device))
        if isinstance(obj, tuple):
            new_obj = tuple(new_obj)
        return new_obj
    elif isinstance(obj, nn.Module):
        obj = obj.to(device)
        return obj
    return obj
