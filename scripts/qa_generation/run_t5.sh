CUDA_VISIBLE_DEVICES=0,1 python t5.py -t manual -m t5-3b
CUDA_VISIBLE_DEVICES=0,1 python predict.py -x manual -y manual -m t5-3b
CUDA_VISIBLE_DEVICES=0,1 python predict.py -x manual -y kairos_argument -m t5-3b

CUDA_VISIBLE_DEVICES=0,1 python t5.py -t kairos_argument -m t5-3b
CUDA_VISIBLE_DEVICES=0,1 python predict.py -x kairos_argument -y manual -m t5-3b
CUDA_VISIBLE_DEVICES=0,1 python predict.py -x kairos_argument -y kairos_argument -m t5-3b

