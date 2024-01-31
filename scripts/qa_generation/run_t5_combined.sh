CUDA_VISIBLE_DEVICES=2,3 python t5.py -t manual-kairos_argument -m t5-large
CUDA_VISIBLE_DEVICES=2,3 python predict.py -x manual-kairos_argument -y manual -m t5-large
CUDA_VISIBLE_DEVICES=2,3 python predict.py -x manual-kairos_argument -y kairos_argument -m t5-large

