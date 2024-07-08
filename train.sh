export NGPU=8;
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 10848 train.py \
	--train_data /data/train_output_modified.pkl \
	--eval_data /data/test_output_modified.pkl \
	--use_checkpoint \
	--lr 0.000075 \
	--model_size large \
	--num_workers 8 \
	--optim adamw \
	--box_number 36 \
	--scheduler linear \
	--weight_decay 0.01 \
	--save_freq 5000 \
	--eval_freq 500 \
	--print_freq 500 \
	--text_maxlength 256 \
	--seed 1357 \
	--name exp-test-large-1357 \
	--checkpoint_dir /data/checkpoints \
	--per_gpu_batch_size 1 \
	--n_block 9 \
	--n_tags 30 \
	--n_im_context 5 \
	--n_ex_context 40 \
	--total_step 10000 \
	--warmup_step 1000 

