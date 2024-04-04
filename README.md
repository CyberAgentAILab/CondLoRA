# CondLoRA

## How to use

### Environment
- OS: Debian GNU/Linux 10 (buster)
- Python version: 3.8.5
- CUDA version: 11.3.109

### Install libraries

```
pip install -r requirements.txt
cd ConditionalLoRA
python setup.py install
```

### Hyperparameter (learning rate) search 

```
cd experiments
bash hp_search.sh [GPU number] [GLEU task name (e.g. sst2, mnli, etc.)] [lora type (e.g. lora, adalora, conditional_lora)] [seed number]
```

### Train Model

```
cd experiments/[GLEU task name (e.g. sst2, mnli, etc.)]
bash train.sh [GPU number] [lora type (e.g. lora, adalora, conditional_lora)] [seed number]
```

### Evaluate Model

```
python src/evaluate.py --model_path [path to trained lora model] --task [GLEU task name (e.g. sst2, mnli, etc.)] --batch_size [batch size] --max_length [max length of input text]
```
