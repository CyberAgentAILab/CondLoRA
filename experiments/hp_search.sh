GPU=$1
TASK=$2
LORA_TYPE=$3
SEED=$4

ROOT=../
SRC=$ROOT/src
MODEL=roberta-base
EVAL_STEPS=1500 

SHCEDULER=linear
BATCH_SIZE=16
MAX_LENGTH=512
R=8
LORA_ALPHA=8
LORA_DROPOUT=0.0
LORA_X_SCALING=0.5
WR=0.06
WD=0.1

if [ $TASK = "qnli" ] || [ $TASK = "mnli" ]  || [ $TASK = "qqp" ]; then
    MAX_EPOCHS=50
elif [ $TASK = "sst2" ]; then
    MAX_EPOCHS=60
else
    MAX_EPOCHS=100
fi


SAVE_DIR=$ROOT/outputs/$MODEL/$TASK/$LORA_TYPE/hp_search/batch_size=${BATCH_SIZE}-max_length=${MAX_LENGTH}-r=${R}-alpha=${LORA_ALPHA}-xscaling=${LORA_X_SCALING}-dropout=${LORA_DROPOUT}-lr=${LR}-scheduler=${SHCEDULER}-wr=${WR}-wd=${WD}/seed$SEED/
mkdir -p $SAVE_DIR

CUDA_VISIBLE_DEVICES=$GPU python $SRC/train_hp_search.py \
    --model_name_or_path $MODEL \
    --task $TASK \
    --batch_size $BATCH_SIZE \
    --max_epochs $MAX_EPOCHS \
    --evaluation_steps $EVAL_STEPS \
    --save_dir $SAVE_DIR \
    --mode train \
    --max_length $MAX_LENGTH \
    --lr_scheduler_type $SHCEDULER \
    --early_stopping 30 \
    --warmup_ratio $WD \
    --lora_type $LORA_TYPE \
    --r $R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_x_scaling $LORA_X_SCALING \
    --num_cycles 4 > $SAVE_DIR/train.log
