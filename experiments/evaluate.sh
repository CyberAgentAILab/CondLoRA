ROOT=..
SRC=$ROOT/src
GPU=$1
MODEL_PATH=$2
TASK=$3

SAVE_PATH=$MODEL_PATH/evaluate.log
CUDA_VISIBLE_DEVICES=$GPU python $SRC/evaluate.py \
    --model_path $MODEL_PATH \
    --task $TASK \
    --batch_size 16 \
    --max_length 512 \
    --mode valid > $SAVE_PATH