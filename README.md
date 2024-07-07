# Knowledge-based Visual Question Answering - Multimodal Artificial Intelligence Project - Group N

## Getting Started

### Installation
To establish the environment, just run this code in the shell:
```
https://github.com/missalt/fine-tuning-vqa.git
cd fine-tuning-vqa
conda env create -f requirements.yaml
conda activate revive
```
That will create the environment ```revive``` we used.
### Download data
We provide the pre-processed data, it contains visual features,  question-specific captions, 
answer candidates for each sample.

Download the pre-processed data, which contains two files ("train_output.pkl" and "test_output.pkl").
```
pip install gdown
gdown https://drive.google.com/uc?id=12djn3dC7huVllRxeSdOPCSImpSChhZKQ
unzip processed_data.zip
```
It will create a folder data:
```
fine-tuning-vqa
├── ...
├── processed_data
│   ├── train_output.pkl
│   ├── test_output.pkl
└── ...
```
### Pre-trained model
|Model |Description|Accuracy(%)|Weight|Log
|  ----  | ----  | ----  | ---- | ---- | 
|Ours (Single)|large size and trained with visual features, question specific captions and answer candidates| 55.1 |Available upon request (too large)|[run.log](https://drive.google.com/file/d/1qsqh0-xJDKv-ZKMlxq4IV2QSMYrjRBeQ/view?usp=drive_link)|

As for **model ensembling**, you can train three models with different seeds, and for each sample, 
you can get the final result with the **highest occurence frequency** among the three models' predictions,
please refer to **ensemble.py**.

### Prediction results
The prediction results of **"single"** and **"ensemble"** versions are shared:

|Model |Accuracy(%)|Download|
|  ----  | ----  | ---- |  
|Ours (Single)| 55.1 |[prediction_acc56.6.json](https://drive.google.com/file/d/1KjMa-XjWjLIwQBg6JhCoLUtJQ9rIMON-/view?usp=sharing)|
|Ours (Ensemble)| 56.8 |[prediction_acc58.1.json](https://drive.google.com/file/d/1rvIP74bfGP5aLr9x2yMn03_f0KrnG0OH/view?usp=sharing)|


### Train the model
Run the following command to start training :
```
export NGPU=4;
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 10847 train.py \
          --train_data processed_data/train_output.pkl \
          --eval_data processed_data/test_output.pkl \
          --use_checkpoint \
          --lr 0.000075 \
          --model_size large \
          --num_workers 16 \
          --optim adamw \
          --box_number 36 \
          --scheduler linear \
          --weight_decay 0.01 \
          --save_freq 2000 \
          --eval_freq 500 \
          --print_freq 100 \
          --text_maxlength 256 \
          --seed 833 \
          --name exp \
          --checkpoint_dir ./checkpoints \
          --per_gpu_batch_size 1 \
          --n_block 9 \
          --n_tags 30 \
          --n_im_context 5 \
          --n_ex_context 40 \
          --total_step 10000 \
          --warmup_step 1000
```

### Test the trained model
Run the following command to start evaluation:
```
CUDA_VISIBLE_DEVICES=0 python test.py --eval_data processed_data/test_output.pkl \
          --model_size large \
          --per_gpu_batch_size 8 \
          --num_workers 4 \
          --box_number 36 \
          --text_maxlength 256 \
          --n_im_context 5 \
          --n_ex_context 40 \
          --name eval \
          --model_path checkpoints/exp/checkpoint/best_dev/ \
          --n_block 9 \
          --n_tags 30 \
          --write_results
```
          
It will not only output the final accuracy, but also 
generate the final results as "prediction.json" under the defined
checkpoint directory path.

### Test with json file
If your prediction json file is named as: "prediction.json".

Run the following command to start evaluation with json files:
```
python leaderboard_evaluation.py --pred_path prediction.json \
          --gt_path eval/mscoco_val2014_annotations.json
```

## Acknowledgements
Our code strongly adopts [REVIVE](https://github.com/yuanze-lin/REVIVE). We thank the authors for open-sourcing their amazing work.
