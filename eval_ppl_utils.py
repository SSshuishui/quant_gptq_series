import time
import torch
import torch.nn as nn
from categories import subcategories, categories
from lm_eval import evaluator

@torch.no_grad()
def llama_eval_ppl(args, model, testenc, dev, logger):
    logger.info('Evaluating ...')

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

    hf_device_map = model.hf_device_map
    logger.info(hf_device_map)

    for i in range(len(layers)):
        logger.info(f'================={i}==================')
        hf_device = "cuda:0" if len(hf_device_map) == 1 and '' in hf_device_map else f"cuda:{hf_device_map[f'model.layers.{i}']}"
        layer = layers[i].to(hf_device)
        inps = inps.to(hf_device)
        position_ids = position_ids.to(hf_device)
        
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

    hf_device = f"cuda:{hf_device_map[f'model.layers.{len(layers)-1}']}"
    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(hf_device)
    model.lm_head = model.lm_head.to(hf_device)

    testenc = testenc.to(hf_device)
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
    logger.info(ppl.item())

    model.config.use_cache = use_cache


def eval_zero_shot(args, model, tokenizer, logger, task_list, num_fewshot=0,  add_special_tokens=False): 
    import lm_eval
    from lm_eval import utils as lm_eval_utils
    from lm_eval.models.huggingface import HFLM
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmpdir:
        # 1. 把模型瞬时导出
        model.save_pretrained(tmpdir, safe_serialization=True)
        tokenizer.save_pretrained(tmpdir)

        hflm = HFLM(
            pretrained=tmpdir, 
            tokenizer=tokenizer, 
            batch_size=args.lm_eval_batch_size,
            parallelize=True)
        
        ALL_TASKS = ["piqa", "copa", "boolq", "rte", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]
        task_names = lm_eval_utils.pattern_match(task_list, ALL_TASKS)
        print(f"Matched Tasks: {task_names}")
        if not task_names:
            raise ValueError("No tasks matched. Check task names!")

        results = lm_eval.simple_evaluate(
            model=hflm, 
            tasks=task_names, 
            num_fewshot=num_fewshot,
            batch_size=args.lm_eval_batch_size
        )

        metric_vals = process_eval_results(results, logger)
        logger.info(f"{args.wbits} - {args.act_bits} - Evaluation Results: {metric_vals}")
