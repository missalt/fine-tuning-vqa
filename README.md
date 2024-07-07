# Knowledge-based Visual Question Answering - Multimodal Artificial Intelligence Project - Group N

## Getting Started

### Installation
To establish the environment, just run this code in the shell:
```
git clone https://github.com/yzleroy/REVIVE.git
cd REVIVE
conda env create -f requirements.yaml
conda activate revive
```
That will create the environment ```revive``` we used.
### Download data
We provide the pre-processed data, it contains visual features,  implicit/explicit knowledge, 
bounding boxes, caption, tags for each sample.

Download the pre-processed data, which contains two files ("train.pkl" and "test.pkl").
```
pip install gdown
gdown https://drive.google.com/uc?id=1kP_xeuUCAS5wqWQwuwVItDgRTAbEjUeM&export=download
unzip processed_data.zip
```
It will create a folder data:
```
REVIVE
├── ...
├── processed_data
│   ├── train.pkl
│   ├── test.pkl
└── ...
```
Note that in "train.pkl" and "test.pkl" data, the meanings of the keys are explained as following: 

"**im_ctxs**": implicit knowledge, each one among it means one implicit knowledge sample.  
"**ex_ctxs**": explicit knowledge, each one among it means one explicit knowledge sample.  
"**boxes**": detected bounding boxes from the object detector.  
"**vis_feat**": visual features which correspond to the detected bounding boxes.  
"**tags**": retrieved tags according to CLIP similarity.  

### Pre-trained model
|Model |Description|Accuracy(%)|Weight|Log
|  ----  | ----  | ----  | ---- | ---- | 
|REVIVE (Single)|large size and trained with visual features, explicit and implicit knowledge| 56.6 |[model.zip](https://drive.google.com/file/d/1yCEgGaxz-GNR4WS89d8ndvuB9bZmMBy_/view?usp=sharing)|[run.log](https://drive.google.com/file/d/1JaSigxV7UoVN5GvYZe0qdyfzLIczTmo7/view?usp=sharing)|

As for **model ensembling**, you can train three models with different seeds, and for each sample, 
you can get the final result with the **highest occurence frequency** among the three models' predictions,
please refer to **ensemble.py**.

### Prediction results
The prediction results of **"single"** and **"ensemble"** versions are shared:

|Model |Accuracy(%)|Download|
|  ----  | ----  | ---- |  
|REVIVE (Single)| 56.6 |[prediction_acc56.6.json](https://drive.google.com/file/d/1KjMa-XjWjLIwQBg6JhCoLUtJQ9rIMON-/view?usp=sharing)|
|REVIVE (Ensemble)| 58.1 |[prediction_acc58.1.json](https://drive.google.com/file/d/1rvIP74bfGP5aLr9x2yMn03_f0KrnG0OH/view?usp=sharing)|


### Train the model
Run the following command to start training (the A5000 training example):
```
export NGPU=4;
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 10847 train.py \
          --train_data processed_data/train.pkl \
          --eval_data processed_data/test.pkl \
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
The whole training time is about 18 hours with 4 X A5000 GPUs.

### Test the trained model
Run the following command to start evaluation:
```
CUDA_VISIBLE_DEVICES=0 python test.py --eval_data processed_data/test.pkl \
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

## Experimental Results

### Comparison with previous methods

![comparison](https://github.com/yzleroy/REVIVE/blob/main/figures/1.png)

### Example visualization

![visualization](https://github.com/yzleroy/REVIVE/blob/main/figures/2.png)

## Contact
If I cannot timely respond to your questions, you can send the email to yuanze@uw.edu.

## Acknowledgements
Our code is built on [FiD](https://github.com/facebookresearch/FiD) which is under the [LICENSE](https://github.com/facebookresearch/FiD/blob/main/LICENSE).
