
# This project mainly integrates several LLMs quantization methods derived from GPTQ

Include:
| Methods | Quantize | PPL Eval | Task Eval | Save |
| :--- | ---: | :---: | :---: | :---: 
| [GPTQ](https://arxiv.org/pdf/2210.17323) | yes | yes | TODO | TODO 
| [PB-LLM](https://arxiv.org/pdf/2310.00034) | yes | yes | TODO | TODO 
| [BiLLM](https://arxiv.org/pdf/2402.04291) | yes | yes | TODO | TODO 
| [CLAQ](https://arxiv.org/pdf/2405.17233) | yes | yes | TODO | TODO 
| [Z-Fold](https://aclanthology.org/2023.emnlp-main.892.pdf) | yes | yes | TODO | TODO 
| [Slim](https://arxiv.org/pdf/2405.14917) | yes | yes | TODO | TODO 
| [QuIP](https://arxiv.org/pdf/2307.13304) | yes | yes | TODO | TODO 
| AWRQ | yes | yes | TODO | TODO 
| [DecoupleQ](https://arxiv.org/pdf/2404.12759) | yes | yes | TODO | TODO 
| [OWQ](https://arxiv.org/pdf/2306.02272) | yes | yes | TODO | TODO 
| [GPTVQ](https://arxiv.org/pdf/2402.15319) | yes | yes | TODO | TODO 
| [MagR](https://arxiv.org/pdf/2406.00800) | yes | yes | TODO | TODO 
| [SpQR](https://arxiv.org/pdf/2306.03078) | yes | yes | TODO | TODO 

## GPTQ 
```
python llama.py --method gptq --model ${MODEL_DIR} --dataset c4 --wbits 4 --true-sequential --act_order --save
```


## BiLLM
```
python llama.py --method billm --model ${MODEL_DIR} --dataset c4 --low_quant_method braq --blocksize 128 --salient_metric hessian --save
```


## Z-fold
```
python llama.py --method zfold --model ${MODEL_DIR} --dataset c4 --wbits 4 --salient_metric hessian --act_order --use_zfold --save
```


## CLAQ 
```
# Run single-precision quantization and compute PPL results
python llama.py --method claq --model ${MODEL_DIR} --dataset c4 --wbits 4 --true-sequential --act_order  --save

# Run Adaptive Precision quantization to quantize the model to 2.1 bit and compute PPL results
python llama.py --method claq --model ${MODEL_DIR} --dataset c4 --wbits 2 --true-sequential --act_order --outlierorder 2.1 --save

# Run Outlier Reservation quantization to keep 0.07 bit of full-precison outliers and compute PPL results
python llama.py --method claq --model ${MODEL_DIR} --dataset c4 --wbits 2 --true-sequential --act_order --outlier_col_dynamic --save

# Run Adaptive Precision + Outlier Reservation quantization to quantize the model to 2.12 bit and compute PPL results
python llama.py --method claq --model ${MODEL_DIR} --dataset c4 --wbits 2 --true-sequential --act_order --outlierorder 2.05 --outlier_col_dynamic
```


## PB-LLM 
```
python llama.py --method pbllm --model ${MODEL_DIR} --dataset c4 --low_quant_method xnor --low_frac 0.5 --high_bit 8 --salient_metric hessian --save

python llama.py --method pbllm --model ${MODEL_DIR} --dataset c4 --low_quant_method xnor --low_frac 0.8 --high_bit 8 --salient_metric hessian --save

python llama.py --method pbllm --model ${MODEL_DIR} --dataset c4 --low_quant_method xnor --low_frac 0.9 --high_bit 8 --salient_metric hessian --save

python llama.py --method pbllm --model ${MODEL_DIR} --dataset c4 --low_quant_method xnor --low_frac 0.95 --high_bit 8 --salient_metric hessian --save

```


## QuIP
```
# Compute full precision (FP16) results
python llama.py --method quip --model ${MODEL_DIR} --dataset c4
# Run a quantization method with baseline processing
python llama.py --method quip --model ${MODEL_DIR} --dataset c4 --wbits 4 --quant gptq --pre_gptqH --save 
```


## SliM
```
# W2A16G128
python llama.py --method slim --model ${MODEL_DIR} --dataset wikitext2 --wbits 4 --groupsize 128 --tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande

# W3A16G128
python llama.py --method slim --model ${MODEL_DIR} --dataset wikitext2 --wbits 3 --groupsize 128 --tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande
```


## AWRQ
```
# with smoothing
python llama.py --method awrq --model ${MODEL_DIR} --dataset c4 --wbits 4 --act_bits 8 --blocksize 1 --smooth --alpha 0.50 --min 0.01 --tasks piqa,arc_easy,arc_challenge,boolq,copa,storycloze 

# without smoothing
python llama.py --method awrq --model ${MODEL_DIR} --dataset c4 --wbits 4 --act_bits 8 --blocksize 1 --tasks piqa,arc_easy,arc_challenge,boolq,copa,storycloze
```


## DecoupleQ for LLaMA families
```
python llama.py \
--method decoupleQ \
--model ${MODEL_DIR} \
--dataset c4 \
--true_sequential \
--act_order \
--wbits 2 \
--max_iter_num 4 \
--iters_before_round 200 \
--inner_iters_for_round 5 \
--blockwise_minimize_epoch 4 \
--round_fn gptq \
--blockwise_minimize_lr 1.0e-5 \
--train_LN \
--save
```


## OWQ
3.01-bit (3-bit quantization + few FP16 weight columns)
```
python llama.py --method owq --model ${MODEL_DIR} --dataset c4 --wbits 3 --target_bit 3.01
```
4.01-bit (4-bit quantization + few FP16 weight columns)
```
python llama.py --method owq --model ${MODEL_DIR} --dataset c4 --wbits 4 --target_bit 4.01
```


## GPTVQ
```
python llama.py --columns-per-group 256 --use-vq --kmeans-iters 100 --kmeans-init-method mahalanobis --hessian-weighted-lookups --wbits 2 --vq-dim 1 --groupsize 512 $LLAMA2_7B_PATH wikitext2
python llama.py --columns-per-group 256 --use-vq --kmeans-iters 100 --kmeans-init-method mahalanobis --hessian-weighted-lookups --wbits 2 --vq-dim 1 --groupsize 512 --include-m-step $LLAMA2_7B_PATH wikitext2
```


## MagR
```
python llama.py --method magr --model ${MODEL_DIR} --dataset c4 --wbits 4 --magr
```


## SpQR
```
python llama.py \
    --method spqr \
--model ${MODEL_DIR} \
--dataset c4 \
--wbits 4 \
--groupsize 16 \
--qq_scale_bits 3 \
--qq_zero_bits 3 \
--qq_groupsize 16 \
--outlier_threshold=0.2 \
--permutation_order act_order \
--percdamp 1e0 \
--nsamples 128 
```