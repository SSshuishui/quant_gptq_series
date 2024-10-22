
## This project mainly integrates several LLMs quantization methods derived from GPTQ

Include:
| Methods | Quantize | PPL Eval | Task Eval | Save |
| :--- | ---: | :---: | :---: | :---: 
| GPTQ | ✅ | ✅ | TODO | TODO 
| PB-LLM | ✅ | ✅ | TODO | TODO 
| BiLLM | ✅ | ✅ | TODO | TODO 
| CLAQ | ✅ | ✅ | TODO | TODO 
| Z-Fold（llama1,2） | ✅ | ✅ | TODO | TODO 
| Slim | ✅ | ✅ | TODO | TODO 
| QuIP | ✅ | ✅ | TODO | TODO 
| DecoupleQ | TODO | TODO | TODO | TODO 

#### GPTQ for LLaMA families
```
python llama.py --method gptq --model ${MODEL_DIR} --dataset c4 --wbits 4 --true-sequential --act-order --save
```


#### Binarization for LLaMA families
```
python llama.py --method billm --model ${MODEL_DIR} --dataset c4 --low_quant_method braq --blocksize 128 --salient_metric hessian --save
```


#### Z-fold for LLaMA families
```
python llama.py --method zfold --model ${MODEL_DIR} --dataset c4 --wbits 4 --salient_metric hessian --act-order --use_zfold --save
```


#### CLAQ for LLaMA families
```
# Run single-precision quantization and compute PPL results
python llama.py --method claq --model ${MODEL_DIR} --dataset c4 --wbits 4 --true-sequential --act-order  --save

# Run Adaptive Precision quantization to quantize the model to 2.1 bit and compute PPL results
python llama.py --method claq --model ${MODEL_DIR} --dataset c4 --wbits 2 --true-sequential --act-order --outlierorder 2.1 --save

# Run Outlier Reservation quantization to keep 0.07 bit of full-precison outliers and compute PPL results
python llama.py --method claq --model ${MODEL_DIR} --dataset c4 --wbits 2 --true-sequential --act-order --outlier_col_dynamic --save

# Run Adaptive Precision + Outlier Reservation quantization to quantize the model to 2.12 bit and compute PPL results
python llama.py --method claq --model ${MODEL_DIR} --dataset c4 --wbits 2 --true-sequential --act-order --outlierorder 2.05 --outlier_col_dynamic
```


#### PB-LLM for LLaMA families
```
python llama.py --method pbllm --model ${MODEL_DIR} --dataset c4 --low_quant_method xnor --low_frac 0.5 --high_bit 8 --salient_metric hessian --save

python llama.py --method pbllm --model ${MODEL_DIR} --dataset c4 --low_quant_method xnor --low_frac 0.8 --high_bit 8 --salient_metric hessian --save

python llama.py --method pbllm --model ${MODEL_DIR} --dataset c4 --low_quant_method xnor --low_frac 0.9 --high_bit 8 --salient_metric hessian --save

python llama.py --method pbllm --model ${MODEL_DIR} --dataset c4 --low_quant_method xnor --low_frac 0.95 --high_bit 8 --salient_metric hessian --save

```

#### DecoupleQ for LLaMA families
```
python llama.py --method decoupleq --model ${MODEL_DIR} --dataset c4 --true-sequential --act-order --new-eval \
    --wbits 2 \
    --group-size -1 \
    --max-iter-num 4 \
    --iters-before-round 200 \
    --inner-iters-for-round 5 \
    --blockwise-minimize-epoch 4 \
    --round-fn gptq \
    --blockwise-minimize-lr 1.0e-5 \
    --train-LN \
    --save
```


#### QuIP
```
# Compute full precision (FP16) results
python llama.py --method quip --model ${MODEL_DIR} --dataset c4
# Run a quantization method with baseline processing
python llama.py --method quip --model ${MODEL_DIR} --dataset c4 --wbits 4 --quant gptq --pre_gptqH --save 
```


#### SliM
```
# W2A16G128
python llama.py --method slim --model ${MODEL_DIR} --dataset wikitext2 --wbits 4 --groupsize 128 --tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande

# W3A16G128
python llama.py --method slim --model ${MODEL_DIR} --dataset wikitext2 --wbits 3 --groupsize 128 --tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande
```

#### AWRQ
```
# with smoothing
python llama.py --method awrq --model ${MODEL_DIR} --dataset c4 --table_results --wbits 4 --act_bits 8 --blocksize 1 --smooth --alpha 0.50 --min 0.01 --tasks piqa,arc_easy,arc_challenge,boolq,copa,storycloze 

# without smoothing
python llama.py --method awrq --model ${MODEL_DIR} --dataset c4  --table_results --wbits 4 --act_bits 8 --blocksize 1 --tasks piqa,arc_easy,arc_challenge,boolq,copa,storycloze
```