import time

import torch
import torch.nn as nn

from utils.modelutils import *
from gptq.quant import *


def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, device_map='auto', torch_dtype='auto')
    model.seqlen = 2048
    return model


@torch.no_grad() 
def llama_nearest(model, dev):
    print("RTN Quantization ...")
    layers = model.model.layers
    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        for name in subset:
            quantizer = Quantizer()
            quantizer.configure(args.wbits, perchannel=True, sym=False, mse=False)
            W = subset[name].weight.data
            quantizer.find_params(W, weight=True)
            subset[name].weight.data = (
                quantize(W, quantizer.scale, quantizer.zero, quantizer.maxq).to(next(iter(layer.parameters())).dtype).view(W.shape)
            )

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
    return model


@torch.no_grad()
def llama_sequential_gptq(model, dataloader, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']

            raise ValueError

    layers[0] = Catcher(layers[0])

    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    quantizers = {}
    hf_device_map = model.hf_device_map
    print(hf_device_map)
        
    for i in range(len(layers)):
        print(f'================={i}==================')
        # layer = layers[i].to(dev)
        layer = layers[i].to(hf_device_map[f'model.layers.{i}'])
        inps = inps.to(hf_device_map[f'model.layers.{i}'])
        position_ids = position_ids.to(hf_device_map[f'model.layers.{i}'])

        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj'],
            ]
        else:
            sequential = [list(full.keys())]
       
        for names in sequential:
            subset = {n: full[n] for n in names}
            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print('Quantizing ...')
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups
                )
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    
    return quantizers


@torch.no_grad()
def llama_sequential_billm(model, dataloader, dev):
    print("Starting ...")

    for name, module in model.named_modules():
        module.global_name = args.model + name

    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache['position_ids']

    print("Ready.")
    
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        subset = find_layers(layer)

        gptq = {}
        for name in subset:
            braq_quantizer = Binarization(
                subset[name].weight,
                method=args.low_quant_method,
                groupsize=args.groupsize,
            )
            gptq[name] = BRAGPTQ(
                subset[name],
                braq_quantizer,
                salient_metric=args.salient_metric
            )

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gptq:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gptq:
            print(i, name)
            print("Quantizing ...")
            info = gptq[name].fasterquant(
                percdamp=args.percdamp, 
                blocksize=args.groupsize,
            )
            gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache


@torch.no_grad()
def llama_sequential_zfold(model, dataloader, dev, nbits, salient_metric, use_zfold):
    print("Starting ...")

    use_hessian = (salient_metric == 'hessian')
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(torch.float32).to(dev)
    model.model.norm = model.model.norm.to(torch.float32).to(dev)

    layers[0] = layers[0].to(dev)
    dtype = torch.float32

    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])

    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    print("Ready.")

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        layer = layer.to(torch.float32)
        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
                ["self_attn.o_proj"],
                ["mlp.up_proj", "mlp.gate_proj"],
                ["mlp.down_proj"],
            ]
        else:
            sequential = [list(full.keys())]

        toggle_share_qkv = False
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            if use_zfold and not toggle_share_qkv:
                tick = time.time()  # additional spending times for Z-fold
                H = gptq["self_attn.q_proj"].H
                dead = torch.diag(H) == 0
                H[dead, dead] = 1
                percdamp = 0.01
                damp = percdamp * torch.mean(torch.diag(H))
                diag = torch.arange(gptq["self_attn.q_proj"].columns, device="cuda")
                H[diag, diag] += damp

                # zfold share QKV
                share_list = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
                qkv_weight = torch.cat([subset[name].weight.data.float() for name in share_list], dim=0)
                qkv_scale, qkv_zfold, qkv_zero, maxq, diff, alternating_iter = find_qkv_params(use_hessian, qkv_weight, nbits, H)
                (
                    gptq["self_attn.q_proj"].quantizer.scale,
                    gptq["self_attn.k_proj"].quantizer.scale,
                    gptq["self_attn.v_proj"].quantizer.scale,
                ) = qkv_scale.view(3, qkv_scale.shape[0] // 3, 1)
                (
                    gptq["self_attn.q_proj"].quantizer.zero,
                    gptq["self_attn.k_proj"].quantizer.zero,
                    gptq["self_attn.v_proj"].quantizer.zero,
                ) = qkv_zero.view(3, qkv_zero.shape[0] // 3, 1)
                for name in share_list:
                    gptq[name].quantizer.scale = gptq[name].quantizer.scale.unsqueeze(0)
                    gptq[name].quantizer.zero = gptq[name].quantizer.zero.unsqueeze(0)
                    gptq[name].quantizer.zeta = qkv_zfold.unsqueeze(1)
                    gptq[name].quantizer.maxq = maxq
                toggle_share_qkv = True
                print("+---------------------------+------------------------+---------+----------------+")
                print("|           Layer           |   delta_W@H@delta_W.T  |   time  | alternaint iter|")
                print("+===========================+=========================+===========+=========+")
                print(f"|{i}: QKV Share          | {diff:.3f}\t| {(time.time() - tick):.2f}\t| {alternating_iter}\t|")

            for name in subset:
                if use_zfold:
                    if name in ["self_attn.k_proj", "self_attn.q_proj", "self_attn.v_proj"]:  # share zeta
                        gptq[name].fasterquant(
                            percdamp=args.percdamp,
                            groupsize=args.groupsize,
                            actorder=args.act_order,
                            static_groups=args.static_groups,
                            ith=i,
                            name=name,
                            use_hessian=use_hessian,
                            use_zfold=use_zfold,
                            share_zeta=True,
                        )
                        quantizers["model.layers.%d.%s" % (i, name)] = gptq[name].quantizer
                    else:
                        if name in ["self_attn.o_proj", "mlp.down_proj"]:  # zfold
                            gptq[name].fasterquant(
                                percdamp=args.percdamp,
                                groupsize=args.groupsize,
                                actorder=args.act_order,
                                static_groups=args.static_groups,
                                ith=i,
                                name=name,
                                use_hessian=use_hessian,
                                use_zfold=use_zfold,
                                share_zeta=False,
                            )
                            quantizers["model.layers.%d.%s" % (i, name)] = gptq[name].quantizer
                        else:
                            gptq[name].fasterquant(
                                percdamp=args.percdamp,
                                groupsize=args.groupsize,
                                actorder=args.act_order,
                                static_groups=args.static_groups,
                                ith=i,
                                name=name,
                                use_hessian=use_hessian,
                                use_zfold=False,
                                share_zeta=False,
                            )
                            quantizers["model.layers.%d.%s" % (i, name)] = gptq[name].quantizer
                else:
                    gptq[name].fasterquant(
                        percdamp=args.percdamp,
                        groupsize=args.groupsize,
                        actorder=args.act_order,
                        static_groups=args.static_groups,
                        ith=i,
                        name=name,
                        use_hessian=use_hessian,
                        use_zfold=False,
                        share_zeta=False,
                    )
                    quantizers["model.layers.%d.%s" % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        inps, outs = outs, inps
        layer = layer.to(torch.float16)
        del layer
        del gptq
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    model.model.embed_tokens = model.model.embed_tokens.to(torch.float16)
    model.model.norm = model.model.norm.to(torch.float16)
    return quantizers


@torch.no_grad()
def llama_sequential_claq(model, dataloader, dev, mix_dict, outlier, outlier_col_dynamic, outlier_layer_dynamic, outlierorder, inputhes):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj']
            ]
        else:
            sequential = [list(full.keys())]
       
        for names in sequential:
            subset = {n: full[n] for n in names}

            claq = {}
            for name in subset:
                claq[name] = CLAQ(subset[name])
                claq[name].quantizer = Quantizer()
                claq[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    claq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print('Quantizing ...')
                layername = '.'+str(i)+'.'+name
                claq[name].fasterquant(args.wbits, layername, outlier, outlier_col_dynamic, outlier_layer_dynamic, outlierorder, inputhes, args.save, percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order)
                quantizers['model.layers.%d.%s' % (i, name)] = claq[name].quantizer
                claq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del claq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    
    return quantizers


@torch.no_grad()
def llama_sequential_decoupleq(args, model, layers, dataloader, dev):
    print("Starting ...")
    cache = []

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            inputs = [list(args), kwargs]
            cache.append(to_device(inputs, "cpu"))
            raise ValueError

    layers[0] = Catcher(layers[0])

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)

    # model = model.to(dev)
    model.eval()
    torch.cuda.empty_cache()
    model.requires_grad_(False)
    masks = [None] * len(dataloader)

    for batch in dataloader:
        batch = to_device(batch, dev)
        try:
            model(batch)
        except ValueError:
            pass

    del dataloader, batch
    gc.collect()
    layers[0] = layers[0].module
    model = model.cpu()
    inps = cache
    torch.cuda.empty_cache()

    print('Ready.')
    shift = 0
    quantizers = {}
    outs = []

    for i in range(len(layers)):
        t_layer0 = time.time()
        layer = layers[i]
        full = find_layers(layer)
        if args.true_sequential:
            sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj']
            ]
        else:
            sequential = [list(full.keys())]

        for k, names in enumerate(sequential):
            subset = {n: full[n] for n in names}
            moq = {}
            for name in subset:
                moq[name] = decoupleQ(subset[name], name=f"layer.{i}.{name}")
                moq[name].quantizer = Quantizer()
                moq[name].quantizer.configure(args.qbits, perchannel=True, sym=not args.asym)
                subset[name].mask = [None]

            def add_batch(name):
                def tmp(module, inp, out):
                    moq[name].add_batch(inp[0].data, out.data, module.mask[0])

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            layer = layer.to(dev)
            for idx, b in enumerate(inps):
                b = to_device(b, dev)
                out = layer(*(b[0]), **b[1])
                if k == 0 and args.blockwise_minimize_lr > 0:
                    os.makedirs("./tmp_blockwise", exist_ok=True)
                    out = {"out": to_device(out, "cpu")}
                    torch.save(out, f"./tmp_blockwise/out_{idx}.pth")
                del out
            layer = layer.cpu()

            for h in handles:
                h.remove()

            for name in names:
                del subset[name].mask
                print(i, name)
                print('Quantizing ...')
                t1 = time.time()
                torch.cuda.empty_cache()
                scale_out, zero_out, w_int, loss = moq[name].startquant(
                    dev=dev,
                    groupsize=args.group_size,
                    symmetric=not args.asym,
                    max_iter_num=args.max_iter_num,
                    inner_iters_for_round=args.inner_iters_for_round,
                    iters_before_round=args.iters_before_round,
                    lr=args.lr,
                    actorder=args.act_order,
                    round_fn=args.round_fn,
                )
                t2 = time.time()
                print(
                    f"time cost {t2 - t1}, model.decoder.layers.{i + shift}.{name}.weight, loss is {loss.mean().item()}")
                print()
                scale_list = [k.cpu() for k in [scale_out, zero_out]]
                quantizers[f"{i + shift}.{name}.weight"] = {
                    "scales": scale_list, "weights": w_int.cpu(), "loss": loss.cpu()}
                moq[name].free()
                moq[name].quantizer.free()
                del moq[name], scale_out, zero_out, w_int
        outs = []
        if args.blockwise_minimize_lr > 0:
            t1 = time.time()
            minimize_block(args, quantizers, layer, inps, dev, i + shift, masks)
            shutil.rmtree("./tmp_blockwise")
            print("time cost for block minimization:", time.time() - t1)

        layer = layer.to(dev)
        for b in inps:
            b = to_device(b, dev)
            outs.append(to_device(layer(*(b[0]), **b[1]), "cpu"))

        layers[i] = layer.cpu()
        del layer
        del moq
        torch.cuda.empty_cache()

        for j in range(len(outs)):
            inps[j][0][0] = outs[j][0]
        del outs
        print(f"quant layer {i} done! time cost {time.time() - t_layer0}")
        print()
    del inps
    model.config.use_cache = use_cache
    return quantizers

@torch.no_grad()
def llama_sequential_pbllm(model, dataloader, dev):
    print("Starting ...")

    for name, module in model.named_modules():
        module.global_name = args.model + name

    use_cache = model.config.use_cache
    model.config.use_cache = False

    if "opt" in args.model:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
            dev
        )
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    elif "llama" in args.model:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    if "opt" in args.model:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif "llama" in args.model:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    print("Ready.")
    plt_x = []
    plt_error = []
    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            if (
                not (args.minlayer <= i < args.maxlayer and args.quant_only in name)
            ) == (not args.invert):
                continue
            low_quantizer = LowQuantizer(
                subset[name].weight,
                method=args.low_quant_method,
                groupsize=args.groupsize,
            )
            high_quantizer = HighQuantizer(
                args.high_bit,
                True,
                False,
                False,
            )
            gpts[name] = LowHighGPT(
                subset[name],
                low_quantizer,
                high_quantizer,
                salient_metric=args.salient_metric,
                disable_gptq=args.disable_gptq,
            )

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print("Quantizing ...")
            info = gpts[name].fasterquant(
                args.low_frac, percdamp=args.percdamp, blocksize=args.blocksize
            )
            gpts[name].free()
            plt_x.append(f"{i}_{name}")
            plt_error.append(info["error"])

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        del gpts
        torch.cuda.empty_cache()

        inps, outs = outs, inps
    if args.plot:
        title = f"{args.model}_{args.dataset}_{args.low_quant_method}_{args.low_frac}_{args.high_bit}"
        torch.save([plt_x, plt_error], "../output/" + title.replace("/", "_") + ".pkl")
        import matplotlib.pyplot as plt

        plt.plot(plt_error)
        plt.xticks(range(1, len(plt_x) + 1), plt_x)
        plt.title(title)
        plt.savefig("../output/" + title.replace("/", "_") + ".jpg")

    model.config.use_cache = use_cache


@torch.no_grad()
def llama_eval(model, testenc, dev):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)
        
        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                quantizer = Quantizer()
                quantizer.configure(
                    args.wbits, perchannel=True, sym=False, mse=False
                )
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantize(
                    W, quantizer.scale, quantizer.zero, quantizer.maxq
                ).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache

def llama_pack3(model, quantizers):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers)
    qlayers = find_layers(model, [Quant3Linear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)
    print('Done.')
    return model

@torch.no_grad()
def z_folding(model, quantizers):
    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i].to("cuda")
        subset = find_layers(layer)
        for name in subset:
            print(i, name)
            # LayerNorm Folding
            if name in ["self_attn.k_proj", "self_attn.q_proj", "self_attn.v_proj"]:
                subset[name].weight.data.div_(quantizers[f"model.layers.{i}.{name}"].zeta)
            # Linear-Layer Folding
            if name == "self_attn.o_proj":
                subset[name].weight.data.div_(quantizers[f"model.layers.{i}.{name}"].zeta)
                subset["self_attn.v_proj"].weight.data.mul_(quantizers[f"model.layers.{i}.{name}"].zeta.T)
            if name == "mlp.down_proj":
                subset[name].weight.data.div_(quantizers[f"model.layers.{i}.{name}"].zeta)
                subset["mlp.up_proj"].weight.data.mul_(quantizers[f"model.layers.{i}.{name}"].zeta.T)
        # LayerNorm Folding
        layer.input_layernorm.weight.data.mul_(quantizers[f"model.layers.{i}.self_attn.q_proj"].zeta.squeeze())


if __name__ == '__main__':
    import argparse
    from utils.datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--method', type=str,
        help='quantization methods to use.'
    )
    parser.add_argument(
        '--model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        '--dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        "--low_quant_method",
        type=str,
        choices=["xnor", "sign", "no", "2bit", "4bit", "prune", "braq"],
        nargs='?',  # 使得参数可选
        const="no",  # 如果用户没有提供参数，则使用这个值
        help="quantization method; `xnor` is the method using XNOR to adapt hardware calculation; `prune` is the method used in sparseGPTQ; braq is the method used in BiLLM",
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        "--salient_metric",
        type=str,
        default="magnitude",
        choices=["magnitude", "hessian"],
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    ) 
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--save', action='store_true',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--true-sequential', action='store_true',
        help='Whether to run in true sequential model.'
    )
    parser.add_argument(
        '--static-groups', action='store_true',
        help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.'
    )

    # For CLAQ args
    parser.add_argument("--outlier", type=float, default=0, help="Max and Min percentage of outliers keeped.")
    parser.add_argument("--inputhes", type=float, default=0, help="Max and Min percentage of outliers keeped.")
    parser.add_argument("--outlierorder", type=float, default=0, help="Use outliers to perform colomn-wise mixed-precision quantization.")
    parser.add_argument("--outlier_col_dynamic", action='store_true', help="outlier colomn dynamic.")
    parser.add_argument("--outlier_layer_dynamic", action='store_true', help="outlier layer dynamic.")

    # For PB-LLM args
    parser.add_argument("--low_frac", type=float, default=0, help="Target low frac.")
    parser.add_argument("--high_bit", type=float, default=0, help="Max bit.")
    parser.add_argument("--plot", action="store_true")

    args = parser.parse_args()

    model = get_llama(args.model)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    if args.method == 'gptq' and args.wbits < 16 and not args.nearest:
        from gptq.gptq import *

        tick = time.time()
        quantizers = llama_sequential_gptq(model, dataloader, DEV)
        print(time.time() - tick)
        if args.save:
            save_title = f"dataset_{args.dataset}_{args.method}_wbits{args.wbits}_seed{args.seed}"
            save_file = "./qmodel/" + save_title + ".pt"
            llama_pack3(model, quantizers)
            torch.save(model.state_dict(), args.save)
   
    elif args.method == 'billm':
        from bigptq import BRAGPTQ
        from binary import Binarization 

        tick = time.time()
        quantizers = llama_sequential_billm(model, dataloader, DEV)
        print(time.time() - tick)
        if args.save:
            save_title = f"dataset_{args.dataset}_{args.method}_lq_method{args.low_quant_method}_groupsz{groupsize}_wbits{args.wbits}_salient_{args.salient_metric}_seed{args.seed}"
            save_file = "./qmodel/" + save_title + ".pt"
            llama_pack3(model, quantizers)
            torch.save(model.state_dict(), args.save)

    elif args.method == 'zfold' and args.wbits < 16 and not args.nearest:
        from gptq_zfold import *

        tick = time.time()
        quantizers = llama_sequential_zfold(model, dataloader, DEV, args.wbits, args.salient_metric, args.use_zfold)
        print(time.time() - tick)
        if args.use_zfold:
            z_folding(model, quantizers)
        if args.save:
            model.save_pretrained(
                f"./qmodel/{args.model}-W{args.wbits}-actorder_{args.act_order}-seed_{args.seed}-zfold_{args.use_zfold}-h_{args.salient_metric}"
            )
            torch.save(
                quantizers,
                f"./qmodel/{args.model}-W{args.wbits}-actorder_{args.act_order}-seed_{args.seed}-zfold_{args.use_zfold}-h_{args.salient_metric}/q_params.pt",
            )

    elif args.method == 'claq' and args.wbits < 16 and not args.nearest:
        tick = time.time()
        quantizers = llama_sequential_claq(model, dataloader, DEV, args.outlier, args.outlier_col_dynamic, args.outlier_layer_dynamic, args.outlierorder, args.inputhes)
        print(time.time() - tick)
        if args.save:
            save_title = f"dataset_{args.dataset}_{args.method}_wbits{args.wbits}_seed{args.seed}"
            save_file = "./qmodel/" + save_title + ".pt"
            llama_pack3(model, quantizers)
            torch.save(model.state_dict(), args.save)

    elif args.method == 'slim':
        pass
    
    elif args.method == 'decoupleQ':
        tick = time.time()
        quantizers = llama_sequential_decoupleq(args, model, layers, dataloader, dev=dev)
        print(time.time() - tick)
        if args.save:
            save_title = f"dataset_{args.dataset}_{args.method}_wbits{args.wbits}_seed{args.seed}"
            save_file = "./qmodel/" + save_title + ".pt"
            llama_pack3(model, quantizers)
            torch.save(model.state_dict(), args.save)
    
    elif args.method == 'pbllm':
        from gptq.pbllm import *
        tick = time.time()
        llama_sequential_pbllm(model, dataloader, device)
        print(time.time() - tick)
        if args.save:
            save_title = f"dataset_{args.dataset}_{args.method}_wbits{args.wbits}_seed{args.seed}"
            save_file = "./qmodel/" + save_title + ".pt"
            llama_pack3(model, quantizers)
            torch.save(model.state_dict(), args.save)
    
    elif args.method == 'quip':
        pass


    for dataset in ['wikitext2', 'ptb', 'c4']:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        llama_eval(model, testloader, DEV)
    