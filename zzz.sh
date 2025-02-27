export HYDRA_FULL_ERROR=1

python train.py --config-name='train_dp_moe_rgb' \
        task=rlbench/mt4 \
        training.num_epochs=2001 \
        training.num_demo=200 \
        dataloader.batch_size=256 \
        val_dataloader.batch_size=256 \
        dataloader.num_workers=4 \
        val_dataloader.num_workers=4 \
        training.rollout_every=250 \
        training.checkpoint_every=10 \
        training.val_every=1 \
        training.sample_every=1 \
        task.lazy_eval=True