GPU=$1
LORA_TYPE=$2
SEED=$3
ROOT=../../
SRC=$ROOT/src
MODEL=roberta-base
TASK=cola
EVAL_STEPS=1500

BATCH_SIZE=16
MAX_LENGTH=512
R=8
LORA_ALPHA=8
LORA_DROPOUT=0.1
LORA_X_SCALING=0.5
LR=4e-4
SHCEDULER=linear
WR=0.06
WD=0.1

SAVE_DIR=$ROOT/outputs/$MODEL/$TASK/$LORA_TYPE/batch_size=${BATCH_SIZE}-max_length=${MAX_LENGTH}-r=${R}-alpha=${LORA_ALPHA}-xscaling=${LORA_X_SCALING}-dropout=${LORA_DROPOUT}-lr=${LR}-wr=${WR}-wd=${WD}-lorasetting/seed$SEED/
mkdir -p $SAVE_DIR

CUDA_VISIBLE_DEVICES=$GPU python $SRC/train.py \
    --model_name_or_path $MODEL \
    --task $TASK \
    --batch_size $BATCH_SIZE \
    --max_epochs 80 \
    --evaluation_steps $EVAL_STEPS \
    --save_dir $SAVE_DIR \
    --mode train \
    --max_length $MAX_LENGTH \
    --learning_rate $LR \
    --lr_scheduler_type $SHCEDULER \
    --warmup_ratio $WD \
    --lora_type $LORA_TYPE \
    --r $R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_x_scaling $LORA_X_SCALING \
    --setting lora > $SAVE_DIR/train.log
