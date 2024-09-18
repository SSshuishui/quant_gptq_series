python llama.py --method zfold --model /data/BaseLLMs/llama3-8b --dataset c4 --wbits 4 --salient_metric hessian --act-order --use_zfold

python llama.py --method claq --model /data/BaseLLMs/llama3-8b --dataset c4 --wbits 2 --true-sequential --act-order --outlierorder 2.05 --outlier_col_dynamic

python llama.py --method slim --model /data/BaseLLMs/llama3-8b --dataset wikitext2 --wbits 4 --groupsize 128