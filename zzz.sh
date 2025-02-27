export HYDRA_FULL_ERROR=1

python train.py --config-name='train_dp_moe_rgb' \
        task=rlbench/mt4 \
        training.num_epochs=2001 \
        training.num_demo=200 \
        dataloader.batch_size=32 \
        val_dataloader.batch_size=32 \
        dataloader.num_workers=4 \
        val_dataloader.num_workers=4 \
        training.rollout_every=10 \
        training.checkpoint_every=1 \
        training.val_every=1 \
        training.sample_every=1 \
        training.max_train_steps=5 \
        training.max_val_steps=5 \
        task.lazy_eval=True \
        # hydra.run.dir='output/20250227/124756_train_dp_moe_rgb_rb-mt4_N200'