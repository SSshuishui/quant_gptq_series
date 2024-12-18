import time
import torch
import torch.nn as nn
from utils.modelutils import *
from utils.logutils import create_logger
from pathlib import Path


def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    if "opt" in model:
        from transformers import OPTForCausalLM
        model = OPTForCausalLM.from_pretrained(model, torch_dtype="auto")
        model.seqlen = model.config.max_position_embeddings
    elif "llama" in model:
        from transformers import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(model, device_map='auto', torch_dtype='auto')
        # model = LlamaForCausalLM.from_pretrained(model, device_map='cpu', torch_dtype='auto')
        model.seqlen = 2048
    return model


@torch.no_grad() 
def llama_nearest(model, dev):
    logger.info("RTN Quantization ...")
    layers = model.model.layers
    for i in range(len(layers)):
        logger.info(i)
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
    logger.info('Starting ...')

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
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None}

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

    logger.info('Ready.')

    quantizers = {}
    hf_device_map = model.hf_device_map
    logger.info(hf_device_map)
        
    for i in range(len(layers)):
        logger.info(f'================={i}==================')
        hf_device = f"cuda:{hf_device_map[f'model.layers.{i}']}"
        layer = layers[i].to(hf_device)
        inps = inps.to(hf_device)
        position_ids = position_ids.to(hf_device)

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
                logger.info('Quantizing ...')
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
    logger.info("Starting ...")

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
    cache = {"i": 0, "attention_mask": None, 'position_ids': None}

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

    logger.info("Ready.")
    hf_device_map = model.hf_device_map
    logger.info(hf_device_map)
    
    for i in range(len(layers)):
        logger.info(f'================={i}==================')
        hf_device = f"cuda:{hf_device_map[f'model.layers.{i}']}"
        layer = layers[i].to(hf_device)
        inps = inps.to(hf_device)
        position_ids = position_ids.to(hf_device)

        subset = find_layers(layer)
        gptq = {}
        for name in subset:
            braq_quantizer = Binarization(
                subset[name].weight,
                method=args.low_quant_method,
                groupsize=args.blocksize,
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
            logger.info("Quantizing ...")
            info = gptq[name].fasterquant(
                percdamp=args.percdamp, 
                blocksize=args.blocksize,
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
    logger.info("Starting ...")

    use_hessian = (salient_metric == 'hessian')
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(torch.float32).to(dev)
    model.model.norm = model.model.norm.to(torch.float32).to(dev)

    layers[0] = layers[0].to(dev)
    dtype = torch.float32

    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None, 'position_ids': None}

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

    logger.info("Ready.")
    hf_device_map = model.hf_device_map
    logger.info(hf_device_map)

    quantizers = {}
    for i in range(len(layers)):
        logger.info(f'================={i}==================')
        hf_device = f"cuda:{hf_device_map[f'model.layers.{i}']}"
        layer = layers[i].to(hf_device)
        inps = inps.to(hf_device)
        position_ids = position_ids.to(hf_device)

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
                gptq[name].quantizer = ZFoldQuantizer()
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
                diag = torch.arange(gptq["self_attn.q_proj"].columns, device=f"cuda:{hf_device_map[f'model.layers.{i}']}")
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
                logger.info("+---------------------------+------------------------+---------+----------------+")
                logger.info("|           Layer           |   delta_W@H@delta_W.T  |   time  | alternaint iter|")
                logger.info("+===========================+=========================+===========+=========+")
                logger.info(f"|{i}: QKV Share          | {diff:.3f}\t| {(time.time() - tick):.2f}\t| {alternating_iter}\t|")

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
def llama_sequential_claq(model, dataloader, dev):
    logger.info('Starting ...')

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

    logger.info('Ready.')
    hf_device_map = model.hf_device_map
    logger.info(hf_device_map)

    quantizers = {}
    for i in range(len(layers)):
        logger.info(f'================={i}==================')
        hf_device = f"cuda:{hf_device_map[f'model.layers.{i}']}"
        layer = layers[i].to(hf_device)
        inps = inps.to(hf_device)
        position_ids = position_ids.to(hf_device)

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
                claq[name].quantizer = CLAQQuantizer()
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
                logger.info('Quantizing ...')
                layername = '.'+str(i)+'.'+name
                claq[name].fasterquant(args.wbits, layername, args.outlier, args.outlier_col_dynamic, args.outlier_layer_dynamic, args.outlierorder, args.inputhes, save_quant=args.save, blocksize=args.blocksize, percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order)
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
def llama_sequential_pbllm(model, dataloader, dev):
    logger.info("Starting ...")

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
    cache = {"i": 0, "attention_mask": None, 'position_ids': None}

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
    position_ids = cache["position_ids"]

    logger.info("Ready.")
    plt_x = []
    plt_error = []

    hf_device_map = model.hf_device_map
    logger.info(hf_device_map)

    for i in range(len(layers)):
        logger.info(f'================={i}==================')
        hf_device = f"cuda:{hf_device_map[f'model.layers.{i}']}"
        layer = layers[i].to(hf_device)
        inps = inps.to(hf_device)
        position_ids = position_ids.to(hf_device)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
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
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            logger.info("Quantizing ...")
            info = gpts[name].fasterquant(
                args.low_frac, percdamp=args.percdamp, blocksize=args.blocksize
            )
            gpts[name].free()
            plt_x.append(f"{i}_{name}")
            plt_error.append(info["error"])

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del gpts
        torch.cuda.empty_cache()

        inps, outs = outs, inps
        
    if args.plot:
        title = f"{args.model}_{args.dataset}_{args.low_quant_method}_{args.low_frac}_{args.high_bit}"
        torch.save([plt_x, plt_error], "./outputs/" + title.replace("/", "_") + ".pkl")
        import matplotlib.pyplot as plt

        plt.plot(plt_error)
        plt.xticks(range(1, len(plt_x) + 1), plt_x)
        plt.title(title)
        plt.savefig("./outputs/" + title.replace("/", "_") + ".jpg")

    model.config.use_cache = use_cache

@torch.no_grad()
def llama_sequential_quip(model, dataloader, dev):
    logger.info('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False

    if "opt" in args.model:
        layers = model.model.decoder.layers

        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder,
                'project_out') and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(
                dev)
        if hasattr(model.model.decoder,
                'project_in') and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    elif "llama" in args.model:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)

    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)

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
    if "opt" in args.model:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder,
                'project_out') and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder,
                'project_in') and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif "llama" in args.model:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    logger.info('Ready.')

    quantizers = {}
    errors, Hmags, times = [], [], []

    hf_device_map = model.hf_device_map
    logger.info(hf_device_map)

    for i in range(len(layers)):
        logger.info(f'================={i}==================')
        hf_device = f"cuda:{hf_device_map[f'model.layers.{i}']}"
        layer = layers[i].to(hf_device)
        inps = inps.to(hf_device)
        position_ids = position_ids.to(hf_device)

        subset = find_layers(layer)
        quant_method = {}
        # Initialize Quant Method and Compute H
        for name in subset:
            if args.quant == 'gptq':
                quant_method[name] = QuIP_GPTQ(subset[name])
                quant_method[name].quantizer = Quantizer()
                quant_method[name].quantizer.configure(args.wbits,
                                               perchannel=True,
                                               sym=False,
                                               qfn=args.qfn,
                                               mse=False)

        def add_batch(name):
            def tmp(_, inp, out):
                quant_method[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0),
                            attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
        # (H / nsamples).to(torch.float32)
        for name in subset:
            quant_method[name].post_batch()

        # Quantize Weights
        for name in subset:
            print(i, name)
            logger.info('Quantizing ...')
            quant_method[name].preproc(
                                preproc_gptqH=args.pre_gptqH, percdamp=args.percdamp,
                                preproc_rescale=args.pre_rescale, 
                                preproc_proj=args.pre_proj, preproc_proj_extra=args.pre_proj_extra)
            if args.quant == 'gptq':
                quant_method[name].fasterquant(groupsize=args.groupsize)
                
            quantizers['model.decoder.layers.%d.%s' %(i, name)] = quant_method[name].quantizer

            errors.append(quant_method[name].error)
            times.append(quant_method[name].time)
            Hmags.append(quant_method[name].Hmag)
            quant_method[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0),
                            attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del quant_method
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    logger.info(f'Total quant time: {sum(times):.2f}s')

    return quantizers, errors


@torch.no_grad()
def llama_sequential_slim(model, dataloader, dev, saved_block_precision):
    logger.info("Starting ...")

    for name, module in model.named_modules():
        module.global_name = args.model + name

    use_cache = model.config.use_cache
    model.config.use_cache = False

    if "opt" in args.model:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
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
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None, 'position_ids': None}

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
    position_ids = cache["position_ids"]

    logger.info("Ready.")
    index = 0
    mixed_block_precision = {}
    quantizers = {}
    hf_device_map = model.hf_device_map
    logger.info(hf_device_map)

    for i in range(len(layers)):
        logger.info(f'================={i}==================')
        hf_device = f"cuda:{hf_device_map[f'model.layers.{i}']}"
        layer = layers[i].to(hf_device)
        inps = inps.to(hf_device)
        position_ids = position_ids.to(hf_device)

        subset = find_layers(layer)

        gptq = {}
        for name in subset:
            index += 1
            quantizer = SliM_Quantizer(
                subset[name].weight,
                method=args.wbits,
                groupsize=args.groupsize,
                metric = args.quantizer_metric,
                lambda_salience=args.lambda_salience        
            )
            gptq[name] = SliMGPTQ(
                subset[name],
                quantizer,
                disable_gptq=args.disable_gptq,
                layer_index=index,
                salient_block = args.salient_block,
                nonsalient_block = args.nonsalient_block,
                bit_width = args.wbits
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

        # for Activation Aware mixed-precision blocks determination
        if saved_block_precision is None and args.salient_block == -1:
            for name in gptq:
                gptq[name].get_salience(blocksize=args.groupsize)

            def get_block(name):
                def tmp(_, inp, out):
                    gptq[name].get_block(inp[0].data, out.data, blocksize=args.groupsize)
                return tmp
            handles = []
            for name in gptq:
                handles.append(subset[name].register_forward_hook(get_block(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()
        # end

        mixed_block_precision[i] = {}

        for name in gptq:
            print(i, name)
            logger.info("Quantizing ...")
            layer_block_precision, scales, zeros, g_idx = gptq[name].fasterquant(
                percdamp=args.percdamp, 
                blocksize=args.groupsize,
                layer_name=name,
                saved_block_precision=saved_block_precision[i][name] if (saved_block_precision is not None) else None,
            )
            mixed_block_precision[i][name] = layer_block_precision
            quantizers['model.layers.%d.%s' % (i, name)] = (scales.cpu(), zeros.cpu(), g_idx.cpu(), args.wbits, args.groupsize)

            gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()
        inps, outs = outs, inps
        
    # logger.info("The average bit-width is:  ", sum(mean_bit_width) / len(mean_bit_width), " bits")
    if saved_block_precision is None:
        net = args.model.split("/")[-1]
        save_path = os.path.join(f'./SliM-LLM_group-precision/block_precision_{args.groupsize}_{args.low_quant_method}/', f'{net}.pt')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(mixed_block_precision, save_path)

    model.config.use_cache = use_cache
    return quantizers


def llama_sequential_awrq(model, dataloader, dev):
    logger.info("Starting ...")

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
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    logger.info('Ready.')
    hf_device_map = model.hf_device_map
    logger.info(hf_device_map)

    quantizers = {}
    for i in range(len(layers)):
        logger.info(f'================={i}==================')
        hf_device = f"cuda:{hf_device_map[f'model.layers.{i}']}"
        layer = layers[i].to(hf_device)
        inps = inps.to(hf_device)
        position_ids = position_ids.to(hf_device)

        layer = replace_linear_layer(args.model, layer, smooth=args.smooth, alpha=args.alpha, min=args.min, act_quant=False, act_bits=args.act_bits)

        subset = find_layers(layer)
        awrq = {}
        for name in subset:
            awrq[name] = AWRQ(subset[name], args.method)
            awrq[name].quantizer = AWRQQuantizer()  # quantizer in quant.py
            awrq[name].quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False, trits=True)

        def add_batch(name):
            def tmp(_, inp, out):
                awrq[name].add_batch(inp[0].data, out.data)
            return tmp
        
        # smooth
        if args.smooth:
            smoothing_layer(layer, subset, awrq, inps, attention_mask, position_ids, outs)
            
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
        
        # quantize weights
        for name in subset:
            print(i, name)
            logger.info('Quantizing ...')
            awrq[name].quant_weight(
                percdamp=args.percdamp, blocksize=args.blocksize, groupsize=args.groupsize, actorder=args.act_order
            )
            quantizers['model.layers.%d.%s' % (i, name)] = awrq[name].quantizer
            subset[name].act_quant = True
            awrq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del awrq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    return quantizers


def llama_sequential_gptvq(model, dataloader, dev):
    logger.info('Starting ...')

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

    if args.use_vq:
        QClass = lambda: VQQuantizer(
            vq_dim=args.vq_dim,
            columns_per_group=args.columns_per_group,
            vq_scaling_blocksize=args.vq_scaling_blocksize,
            vq_scaling_norm=args.vq_scaling_norm,
            vq_scaling_n_bits=args.vq_scaling_n_bits,
            vq_scaling_domain=args.vq_scaling_domain,
            kmeans_init_method=args.kmeans_init_method,
            assignment_chunk_size=args.assignment_chunk_size,
            kmeans_iters=args.kmeans_iters,
            codebook_bitwidth=args.codebook_bitwidth,
            quantize_per_codebook=args.quantize_per_codebook,
            quantize_during_kmeans=args.quantize_during_kmeans,
            n_subsample=args.kpp_n_subsample,
        )

    logger.info('Ready.')

    quantizers = {}
    hf_device_map = model.hf_device_map
    logger.info(hf_device_map)
        
    for i in range(len(layers)):
        logger.info(f'================={i}==================')
        hf_device = f"cuda:{hf_device_map[f'model.layers.{i}']}"
        layer = layers[i].to(hf_device)
        inps = inps.to(hf_device)
        position_ids = position_ids.to(hf_device)

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
                gptq[name] = GPTVQ(subset[name])
                gptq[name].quantizer = QClass()
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
                logger.info('Quantizing ...')
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups,
                    include_m_step=args.include_m_step,
                    use_vq=args.use_vq,
                    svd_rank=args.svd_rank,
                    hessian_weighted_lookups=args.hessian_weighted_lookups,
                    only_init_kmeans=args.only_init_kmeans,
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


def llama_sequential_owq(model, meta, dataloader, dev):
    logger.info('Starting ...')

    use_cache = model.config.use_cache
    layers, pre_layers, _ = parsing_layers(model, meta)
    model.config.use_cache = False
    
    for pre_layer in pre_layers:
        pre_layer = pre_layer.to(dev)
    
    layers[0] = layers[0].to(dev)
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {kw:None for kw in meta['inp_kwargs']}
    cache['i'] = 0

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            for key in cache:
                if key == 'i':
                    cache['i'] += 1
                else:
                    cache[key] = kwargs[key]
            raise ValueError
    
    layers[0] = Catcher(layers[0])

    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    for pre_layer in pre_layers:
        pre_layer = pre_layer.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    del cache['i']
    inp_kwargs = cache

    print('Ready.')

    owq_layers = meta['owq_layers']
    ratios = meta['ratios']
    n_out_dict = {l:0 for l in owq_layers.keys()}
    if args.target_bit is not None:
        n_owq_layers = sum(owq_layers.values())
        
        r = (12 / (16 - args.wbits)) * (args.target_bit - args.wbits)
        r /= n_owq_layers

        layer = find_layers(layers[0])
        
        for l in owq_layers:
            # for even number of n_out
            n_out = round(layer[l].weight.data.shape[1] * r * ratios[l])
            if n_out % 2 == 1: n_out += 1
            n_out_dict[l] = n_out
    elif args.target_rank is not None:
        for l in owq_layers:
            n_out_dict[l] = args.target_rank

    quantizers = {}
    hf_device_map = model.hf_device_map
    logger.info(hf_device_map)
        
    for i in range(len(layers)):
        logger.info(f'================={i}==================')
        hf_device = f"cuda:{hf_device_map[f'model.layers.{i}']}"
        layer = layers[i].to(hf_device)
        
        full = find_layers(layer)

        if args.true_sequential:
            sequential = meta['sequential']
        else:
            sequential = [list(full.keys())]
        
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq_owq = {}
            for name in subset:
                gptq_owq[name] = OWQ_GPTQ(subset[name], n_out=n_out_dict[name])
                gptq_owq[name].quantizer = Quantizer(
                    args.wbits, perchannel=True, sym=args.sym, mse=(args.tuning == 'mse')
                )
                gptq_owq[name].quantizer.n_out = n_out_dict[name]
                
            def add_batch(name):
                def tmp(_, inp, out):
                    gptq_owq[name].add_batch(inp[0].data, out.data)
                return tmp
            
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                layer(inps[j].unsqueeze(0), **inp_kwargs)
            for h in handles:
                h.remove()
            
            for name in subset:
                if not args.no_frob_norm:
                    W = subset[name].weight.data.clone().to(torch.float)
                    temp_quantizer = Quantizer(
                        args.wbits, perchannel=True, sym=args.sym, mse=(args.tuning == 'mse')
                    )
                    temp_quantizer.find_params(W, weight=True, num=40)
                    W_quant = temp_quantizer.quantize(W)
                    frob_norm_error = (W - W_quant).pow(2).sum(dim=0)
                else:
                    frob_norm_error = None
                out_ids = gptq_owq[name].hessian_sorting(actorder=args.act_order, frob_norm=frob_norm_error)
                gptq_owq[name].quantizer.out_ids = out_ids
                    
            if not args.no_frob_norm:
                del W
                del W_quant
                del temp_quantizer
                torch.cuda.empty_cache()
            
            for name in subset:
                print(f"Quantizing {meta['prefix']}.{i}.{name}")
                gptq_owq[name].fasterquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order)
                quantizers[f"{meta['prefix']}.{i}.{name}"] = gptq_owq[name].quantizer
                gptq_owq[name].free()
            
        for name in list(full.keys()):
            quantizers[f"{meta['prefix']}.{i}.{name}"] = quantizers[f"{meta['prefix']}.{i}.{name}"].cpu()
            
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **inp_kwargs)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq_owq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache


@torch.no_grad()
def llama_sequential_decoupleq(model, dataloader, dev):
    logger.info("Starting ...")

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
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None}
    
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

    
    model.eval()
    torch.cuda.empty_cache()
    model.requires_grad_(False)
    masks = [None] * len(dataloader)

    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass

    del dataloader, batch
    layers[0] = layers[0].module
    model = model.cpu()

    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    logger.info('Ready.')
    shift = 0
    quantizers = {}
    hf_device_map = model.hf_device_map

    for i in range(len(layers)):
        logger.info(f"================={i}==================")
        hf_device = f"cuda:{hf_device_map[f'model.layers.{i}']}"
        t_layer0 = time.time()
        layer = layers[i].to(hf_device)
        inps = inps.to(hf_device)
        position_ids = position_ids.to(hf_device)

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
                moq[name].quantizer.configure(args.wbits, perchannel=True, sym=not args.sym)
                subset[name].mask = [None]

            def add_batch(name):
                def tmp(module, inp, out):
                    moq[name].add_batch(inp[0].data, out.data, module.mask[0])
                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in names:
                del subset[name].mask
                print(i, name)
                logger.info('Quantizing ...')
                t1 = time.time()
                torch.cuda.empty_cache()
                scale_out, zero_out, w_int, loss = moq[name].startquant(
                    dev=dev,
                    groupsize=args.groupsize,
                    symmetric=not args.sym,
                    max_iter_num=args.max_iter_num,
                    inner_iters_for_round=args.inner_iters_for_round,
                    iters_before_round=args.iters_before_round,
                    lr=args.PGD_lr,
                    actorder=args.act_order,
                    round_fn=args.round_fn,
                )
                t2 = time.time()
                logger.info(f"time cost {t2 - t1}, model.decoder.layers.{i + shift}.{name}.weight, loss is {loss.mean().item()}")

                scale_list = [k.cpu() for k in [scale_out, zero_out]]
                quantizers[f"{i + shift}.{name}.weight"] = {"scales": scale_list, "weights": w_int.cpu(), "loss": loss.cpu()}
                moq[name].free()
                moq[name].quantizer.free()
                del moq[name], scale_out, zero_out, w_int

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del moq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

        logger.info(f"quant layer {i} done! time cost {time.time() - t_layer0}")
        logger.info()
    del inps
    model.config.use_cache = use_cache
    return quantizers

@torch.no_grad()
def llama_sequential_magr(model, dataloader, dev):
    logger.info("Starting ...")

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
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None}
    
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

    logger.info('Ready.')
    start_time = time.time()

    beta = 1
    CD_iter = 1
    if args.wbits == 2:
        beta = 0.8
        CD_iter = 30
    elif args.wbits == 3:
        beta = 0.9

    quantizers = {}
    hf_device_map = model.hf_device_map
    for i in range(len(layers)):
        logger.info(f"================={i}==================")
        hf_device = f"cuda:{hf_device_map[f'model.layers.{i}']}"
        layer = layers[i].to(hf_device)
        inps = inps.to(hf_device)
        position_ids = position_ids.to(hf_device)

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

            gptq = {}
            for name in subset:
                gptq[name] = MagRGPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False, beta=beta,
                )

            def add_batch(name):
                def tmp(module, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in names:
                print(i, name)
                logger.info('Quantizing ...')
                gptq[name].fasterquant(
                    percdamp=args.percdamp, 
                    groupsize=args.groupsize, 
                    actorder=args.act_order, 
                    static_groups=args.static_groups,
                    magr=args.magr, 
                    CD_iter=CD_iter
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
    end_time = time.time()

    print(f'\nTime used for evaluation: {end_time - start_time}\n')




def llama_pack3(model, quantizers):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers)
    qlayers = find_layers(model, [Quant3Linear])
    logger.info('Packing ...')
    for name in qlayers:
        logger.info(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name].cpu(), quantizers[name].scale, quantizers[name].zero)
    logger.info('Done.')
    return model


@torch.no_grad()
def z_folding(model, quantizers):
    layers = model.model.layers
    hf_device_map = model.hf_device_map
    logger.info(hf_device_map)
    for i in range(len(layers)):
        hf_device = f"cuda:{hf_device_map[f'model.layers.{i}']}"
        layer = layers[i].to(hf_device)
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


def smoothing_layer(layer, subset, awrq, inps, attention_mask, position_ids, outs):

    def add_batch_act_scales(name):
        def tmp(_, inp, out):
            awrq[name].add_batch_act_scales(inp[0].data, out.data)
        return tmp

    # generate act scales
    handles = []
    for name in subset:
        handles.append(subset[name].register_forward_hook(add_batch_act_scales(name)))
    for j in range(args.nsamples):
        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0] # hook
    # smooth scales
    qkv_wight_scales = None

    for name in subset:
        subset[name].act_scales = awrq[name].act_scales.clone() # act scales
        # weight scales
        subset[name].weight_scales = subset[name].weight.abs().max(dim=0)[0]
        # qkv weight scales
        if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
            qkv_wight_scales = torch.max(qkv_wight_scales, subset[name].weight_scales) if qkv_wight_scales is not None else subset[name].weight_scales
   
    # absorb smooth_scales: linear
    for name in subset:
        print()
        if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
            subset[name].weight_scales = qkv_wight_scales
        subset[name].smooth_scales = (subset[name].act_scales.pow(subset[name].alpha) / subset[name].weight_scales.pow(1-subset[name].alpha)).clamp(min=subset[name].min)
        subset[name].weight.data *= subset[name].smooth_scales # weight*scales
        subset[name].act_scales = None 
        subset[name].weight_scales = None
        
    # absorb smooth_scales: layer norm 
    layer.input_layernorm.weight.data /= layer.self_attn.k_proj.smooth_scales            
    layer.post_attention_layernorm.weight.data /= layer.mlp.gate_proj.smooth_scales
    layer.self_attn.q_proj.act_smoothed = True
    layer.self_attn.k_proj.act_smoothed = True
    layer.self_attn.v_proj.act_smoothed = True
    layer.mlp.gate_proj.act_smoothed = True

    for h in handles:
        h.remove()


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
    parser.add_argument("--log_dir",
        default="./log/", type=str, help="direction of logging file."
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
        "--salient_metric", type=str, default="magnitude", choices=["magnitude", "hessian"],
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    ) 
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 5, 6, 7, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        "--blocksize", type=int, default=128,
        help="Blocksize to use for adaptive mask selection.",
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
        '--act_order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--true_sequential', action='store_true',
        help='Whether to run in true sequential model.'
    )
    parser.add_argument(
        '--static_groups', action='store_true',
        help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.'
    )
    parser.add_argument("--tasks",  type=str, default="", help="Task datasets Evaluate")

    # For Zfold args
    parser.add_argument("--use_zfold", action='store_true', help="outlier colomn dynamic.")

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
    parser.add_argument("--disable_gptq", action="store_true")

    # For QuIP args
    parser.add_argument("--quant", type=str, default="gptq", help="QuIP Quantization methods")
    parser.add_argument('--pre_gptqH', action='store_true', help='preprocessing')
    parser.add_argument('--pre_rescale', action='store_true', help='preprocessing')
    parser.add_argument('--pre_proj', action='store_true', help='preprocessing')
    parser.add_argument('--pre_proj_extra', type=int, default=0, choices=[0, 1, 2], help='Extra options to control pre_proj step.')
    parser.add_argument('--qfn', type=str, default='a', help='qfn: a is default, b is sym incoherent based')

    # For Slim args
    parser.add_argument("--salient_block", type=int, default=-1)
    parser.add_argument("--nonsalient_block", type=int, default=-1)
    parser.add_argument("--quantizer_metric", type=str, default="mse", help="quantizer parameter determination metric")
    parser.add_argument("--lambda_salience", type=float, default=1, help="Percent of the average Hessian diagonal to use for dampening.")

    # For AWRQ args
    parser.add_argument("--act_bits", type=int, default=8)
    parser.add_argument('--act_sym', default=False, action='store_true', help='bits used for inps quantization')
    parser.add_argument("--smooth", default=False, action='store_true', help='smooth')
    parser.add_argument("--alpha", type=float, default=0.50) # smooth param
    parser.add_argument("--min", type=float, default=0.1) # smooth param

    # For gptvq args
    parser.add_argument("--use-vq", action="store_true", help="If set, use VQ (multi-dim non-uniform) quantization")
    parser.add_argument("--vq-dim", type=int, default=2, help="Dimensionality of VQ (if using)")
    parser.add_argument("--vq-scaling-blocksize", type=int, default=-1, help="VQ scaling block size")
    parser.add_argument("--vq-scaling-n-bits", type=int, default=4, help="VQ scaling bit-width")
    parser.add_argument("--vq-scaling-norm", type=str, default="max", help="VQ scaling norm")
    parser.add_argument("--vq-scaling-domain", type=str, default="log", choices=["log", "linear"], help="VQ scaling domain")
    parser.add_argument("--include-m-step", action="store_true", help="If set, perform an M-step (centroid updating) after GPTQ with VQ")
    parser.add_argument("--columns-per-group", type=int, default=None, help="For group-/blockwise quant: force number of columns each group spans (rest is absorbed in rows)")
    parser.add_argument("--kmeans-init-method", type=str, default="cdf", choices=["cdf", "kpp", "mahalanobis"], help="init method for Kmeans")
    parser.add_argument("--assignment-chunk-size", type=int, default=None, help="Chunk assignment step for better memory management")
    parser.add_argument("--kmeans-iters", type=int, default=10)
    parser.add_argument("--codebook-bitwidth", type=int, default=None, help="Bitwidth for codebook quantization")
    parser.add_argument("--quantize-per-codebook", action="store_true", default=False, help="Quantize codebooks individually (more overhead) or per column block")
    parser.add_argument("--quantize-during-kmeans", action="store_true", default=False, help="Quantize codebooks after every M-step. If not set: only quantize after k-means")
    parser.add_argument("--model-type", choices=["llama", "mistral", "mixtral"], default="llama", help="In case this is a Mistral model (GPTQ layerwise remains the same)",)
    parser.add_argument("--kpp-n-subsample", type=int, default=10000)
    parser.add_argument("--svd-rank", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save model in")
    parser.add_argument("--hessian-weighted-lookups", action="store_true", default=False)
    parser.add_argument("--only-init-kmeans", action="store_true", default=False)

    # For OWQ
    parser.add_argument('--target_bit', type=float, default=None, help='Effctive target bits for OWQ.')
    parser.add_argument('--target_rank', type=int, default=None, help='Number of outlier channels for OWQ.(if --target_bit is not given)')
    parser.add_argument('--tuning', type=str, default='mse', choices=['mse', 'minmax'], help='Method for quantization parameter tuning.')
    parser.add_argument('--no_frob_norm', action='store_true', help='Whether to use Frobenius norm for OWQ.')
    parser.add_argument('--dtype', type=str, default=None, help='Data type of model. Use bfloat16 for falcon model family or llama 65B model')
    parser.add_argument('--layers', nargs='+', type=str, default=None, help='Layers to apply OWQ.')

    # For DecoupleQ
    parser.add_argument('--blockwise_minimize_lr', type=float, default=-1.0, help='the learning rate for block minimization')
    parser.add_argument('--blockwise_minimize_epoch', type=int, default=3, help='the number of epoch for training the float point part')
    parser.add_argument('--blockwise-minimize-wd', type=float, default=1.0e-6, help='the weight decaying rate for block minimization')
    parser.add_argument('--max_iter_num', type=int, default=3, help='The max iter num for the whole loop')
    parser.add_argument('--inner_iters_for_round', type=int, default=50, help='the number of iters for PGD when use first level approximation')
    parser.add_argument('--iters_before_round', type=int, default=0, help='the number of iters before entering PGD when use first level approximation')
    parser.add_argument('--PGD_lr', type=float, default=0.001, help='the learning rate for PGD')
    parser.add_argument('--round_fn', type=str, choices=["gptq", "train"], default="train", help='the quant method')
    parser.add_argument('--train_LN', action='store_true', help='Whether to train the parameters in norm')
    parser.add_argument('--train_bias', action='store_true', help='Whether to train the bias in linear layer')

    # For MagR
    parser.add_argument('--magr', action='store_true', help='Whether to apply the MagR process.')


    args = parser.parse_args()

    # init logger
    args.log_dir = f"{args.log_dir}/{args.method}-{args.model.split('/')[-1]}-w{args.wbits}"
    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    logger = create_logger(log_dir)
    logger.info(args)
    
    model = get_llama(args.model)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )
    logger.info(f"Dataset {args.dataset} Loaded!")

    if args.method == 'gptq' and args.wbits < 16 and not args.nearest:
        from gptq.gptq import GPTQ
        from gptq.quant import Quantizer
        from eval_ppl_utils import llama_eval_ppl

        model = model.to(DEV)

        # tick = time.time()
        # quantizers = llama_sequential_gptq(model, dataloader, DEV)
        # logger.info(time.time() - tick)

        # for dataset in ['wikitext2', 'ptb', 'c4']:
        #     dataloader, testloader = get_loaders(
        #         dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        #     )
        #     logger.info(dataset)
        #     llama_eval_ppl(args, model, testloader, DEV, logger)

        if args.tasks != "":
            from eval_ppl_utils import zeroshot_evaluate
            zeroshot_evaluate(args, model, DEV, logger)

        # if args.save:
        #     save_title = f"dataset_{args.dataset}_{args.method}_wbits{args.wbits}_seed{args.seed}"
        #     save_file = "./qmodel/" + save_title + ".pt"
        #     llama_pack3(model, quantizers)
        #     torch.save(model.state_dict(), save_file)
   
    elif args.method == 'billm':
        from gptq.bigptq import BRAGPTQ
        from gptq.binary import Binarization 
        from gptq.quant import *
        from eval_ppl_utils import llama_eval_ppl

        tick = time.time()
        llama_sequential_billm(model, dataloader, DEV)
        logger.info(time.time() - tick)

        for dataset in ['wikitext2', 'ptb', 'c4']:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            logger.info(dataset)
            llama_eval_ppl(args, model, testloader, DEV, logger)

        if args.save:
            save_title = f"dataset_{args.dataset}_{args.method}_lq_method{args.low_quant_method}_groupsz{groupsize}_wbits{args.wbits}_salient_{args.salient_metric}_seed{args.seed}"
            save_file = "./qmodel/" + save_title + ".pt"
            torch.save(model.state_dict(), save_file)

    elif args.method == 'zfold' and args.wbits < 16 and not args.nearest:
        from gptq.gptq_zfold import GPTQ
        from gptq.quant import ZFoldQuantizer
        from gptq.zfold import *
        from eval_ppl_utils import llama_eval_ppl

        tick = time.time()
        quantizers = llama_sequential_zfold(model, dataloader, DEV, args.wbits, args.salient_metric, args.use_zfold)
        logger.info(time.time() - tick)
        if args.use_zfold:
            z_folding(model, quantizers)

        for dataset in ['wikitext2', 'ptb', 'c4']:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            logger.info(dataset)
            llama_eval_ppl(args, model, testloader, DEV, logger)
        
        if args.save:
            save_title = f"dataset_{args.dataset}_{args.method}_actorder_{args.act_order}_zfold_{args.use_zfold}_wbits{args.wbits}_salient_{args.salient_metric}_seed{args.seed}"
            save_file = "./qmodel/" + save_title + ".pt"
            torch.save(model.state_dict(), save_file)

    elif args.method == 'claq' and args.wbits < 16 and not args.nearest:
        from gptq.claq import CLAQ
        from gptq.claq_quant import *
        from eval_ppl_utils import llama_eval_ppl

        tick = time.time()
        quantizers = llama_sequential_claq(model, dataloader, DEV)
        logger.info(time.time() - tick)

        for dataset in ['wikitext2', 'ptb', 'c4']:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            logger.info(dataset)
            llama_eval_ppl(args, model, testloader, DEV, logger)

        if args.save:
            save_title = f"dataset_{args.dataset}_{args.method}_wbits{args.wbits}_seed{args.seed}"
            save_file = "./qmodel/" + save_title + ".pt"
            torch.save(model.state_dict(), save_file)
    
    elif args.method == 'pbllm':
        from gptq.pbllm import LowHighGPT
        from gptq.quant import LowQuantizer, HighQuantizer
        from eval_ppl_utils import llama_eval_ppl

        tick = time.time()
        llama_sequential_pbllm(model, dataloader, DEV)
        logger.info(time.time() - tick)

        for dataset in ['wikitext2', 'ptb', 'c4']:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            logger.info(dataset)
            llama_eval_ppl(args, model, testloader, DEV, logger)

        if args.save:
            save_title = f"dataset_{args.dataset}_{args.method}_wbits{args.wbits}_lowfeac{args.low_frac}_highbit{args.high_bit}_seed{args.seed}"
            save_file = "./qmodel/" + save_title + ".pt"
            torch.save(model.state_dict(), save_file)
    
    elif args.method == 'quip':
        from gptq.quip.gptq import QuIP_GPTQ
        from gptq.quip.quant import *
        from eval_ppl_utils import llama_eval_ppl

        tick = time.time()
        llama_sequential_quip(model, dataloader, DEV)
        logger.info(time.time() - tick)

        for dataset in ['wikitext2', 'ptb', 'c4']:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            logger.info(dataset)
            llama_eval_ppl(args, model, testloader, DEV, logger)

        if args.save:
            save_title = f"dataset_{args.dataset}_{args.method}_wbits{args.wbits}_seed{args.seed}"
            save_file = "./qmodel/" + save_title + ".pt"
            torch.save(model.state_dict(), save_file)

    elif args.method == 'slim':
        from gptq.slim.slim import SliM_Quantizer
        from gptq.slim.slim_gptq import SliMGPTQ
        from eval_ppl_utils import llama_eval_ppl

        # get the block precision of the model
        # if the block precision does not exist, start Salience-Determined Bit Allocation
        net = args.model.split("/")[-1]
        block_configurations = f'./SliM-LLM_group-precision/block_precision_{args.groupsize}_{args.wbits}bits/{net}.pt'
        if os.path.exists(block_configurations):
            block_precision = torch.load(block_configurations)
        else:
            logger.info(f'Block precisions of {net} does not exist. Start aware!')
            block_precision = None
    
        tick = time.time()
        llama_sequential_slim(model, dataloader, DEV, block_precision)
        logger.info(time.time() - tick)

        for dataset in ['wikitext2', 'ptb', 'c4']:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            logger.info(dataset)
            llama_eval_ppl(args, model, testloader, DEV, logger)
        
        if args.tasks != "":
            from eval_ppl_utils import zeroshot_evaluate
            zeroshot_evaluate(args, model, DEV)

        if args.save:
            save_title = f"dataset_{args.dataset}_{args.method}_wbits{args.wbits}_seed{args.seed}"
            save_file = "./qmodel/" + save_title + ".pt"
            torch.save(model.state_dict(), save_file)
    
    elif args.method == 'awrq':
        from gptq.awrq import *
        from eval_ppl_utils import llama_eval_ppl
        from gptq.awrq_quant import replace_linear_layer, AWRQQuantizer, find_layers

        tick = time.time()
        quantizers = llama_sequential_awrq(model, dataloader, DEV)
        logger.info(time.time() - tick)

        for dataset in ['wikitext2', 'ptb', 'c4']:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            logger.info(dataset)
            llama_eval_ppl(args, model, testloader, DEV, logger)

        if args.save:
            save_title = f"dataset_{args.dataset}_{args.method}_wbits{args.wbits}_seed{args.seed}"
            save_file = "./qmodel/" + save_title + ".pt"
            torch.save(model.state_dict(), save_file)

    elif args.method == 'gptvq' and args.wbits < 16:
        from gptq.gptvq.gptvq import *
        from gptq.gptvq.vq_quant import VQQuantizer
        from eval_ppl_utils import llama_eval_ppl

        tick = time.time()  
        llama_sequential_gptvq(model, dataloader, DEV)
        logger.info(time.time() - tick)

        for dataset in ['wikitext2', 'ptb', 'c4']:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            logger.info(dataset)
            llama_eval_ppl(args, model, testloader, DEV, logger)

        if args.save:
            save_title = f"dataset_{args.dataset}_{args.method}_wbits{args.wbits}_seed{args.seed}"
            save_file = "./qmodel/" + save_title + ".pt"
            torch.save(model.state_dict(), save_file)

    elif args.method == 'decoupleQ':
        from gptq.decoupleQ.quant import decoupleQ, minimize_block
        from gptq.decoupleQ.moq_quant import Quantizer
        from eval_ppl_utils import llama_eval_ppl
        import shutil

        tick = time.time()
        llama_sequential_decoupleq(model, dataloader, DEV)
        logger.info(time.time() - tick)

        for dataset in ['wikitext2', 'ptb', 'c4']:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            logger.info(dataset)
            llama_eval_ppl(args, model, testloader, DEV, logger)

        if args.save:
            save_title = f"dataset_{args.dataset}_{args.method}_wbits{args.wbits}_seed{args.seed}"
            save_file = "./qmodel/" + save_title + ".pt"
            torch.save(model.state_dict(), save_file)

    elif args.method == 'owq':
        from gptq.owq.owq import OWQ_GPTQ
        from gptq.owq.owq_misc import *
        from gptq.owq.owq_quant import Quantizer
        from eval_ppl_utils import llama_eval_ppl

        meta = processing_arguments(args)

        tick = time.time()
        llama_sequential_owq(model, meta, dataloader, DEV)
        logger.info(time.time() - tick)

        for dataset in ['wikitext2', 'ptb', 'c4']:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            logger.info(dataset)
            llama_eval_ppl(args, model, testloader, DEV, logger)

        if args.save:
            save_title = f"dataset_{args.dataset}_{args.method}_wbits{args.wbits}_seed{args.seed}"
            save_file = "./qmodel/" + save_title + ".pt"
            torch.save(model.state_dict(), save_file)

    elif args.method == 'magr':
        from gptq.MagR import MagRGPTQ
        from gptq.quant import Quantizer
        from eval_ppl_utils import llama_eval_ppl

        tick = time.time()
        llama_sequential_magr(model, dataloader, DEV)
        logger.info(time.time() - tick)

        for dataset in ['wikitext2', 'ptb', 'c4']:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            logger.info(dataset)
            llama_eval_ppl(args, model, testloader, DEV, logger)

        if args.save:
            save_title = f"dataset_{args.dataset}_{args.method}_wbits{args.wbits}_seed{args.seed}"
            save_file = "./qmodel/" + save_title + ".pt"
            torch.save(model.state_dict(), save_file)