export HYDRA_FULL_ERROR=1

# python train.py --config-name='train_dp_moe_rgb' \
#         task=rlbench/mt4 \
#         training.num_epochs=2001 \
#         training.num_demo=200 \
#         dataloader.batch_size=64 \
#         val_dataloader.batch_size=64 \
#         dataloader.num_workers=4 \
#         val_dataloader.num_workers=4 \
#         training.rollout_every=250 \
#         training.checkpoint_every=10 \
#         training.val_every=50 \
#         training.sample_every=50 \
#         task.lazy_eval=False \
#         hydra.run.dir='output/20250227/202056_train_dp_moe_rgb_rb-mt4_N200'

# NOTE: run this
python train.py --config-name='train_dp_moe_rgb' \
        task=rlbench/mt4 \
        training.num_epochs=2001 \
        training.num_demo=200 \
        policy.n_layer=32 \
        dataloader.batch_size=64 \
        val_dataloader.batch_size=64 \
        dataloader.num_workers=4 \
        val_dataloader.num_workers=4 \
        training.rollout_every=250 \
        training.checkpoint_every=10 \
        training.val_every=50 \
        training.sample_every=50 \
        task.lazy_eval=False \
        hydra.run.dir='output/20250228/085306_train_dp_moe_rgb_rb-mt4_N200'

# python train.py --config-name='train_dp_moe_rgb' \
#         task=metaworld/mt10 \
#         training.num_epochs=2001 \
#         training.num_demo=100 \
#         dataloader.batch_size=128 \
#         val_dataloader.batch_size=128 \
#         dataloader.num_workers=4 \
#         val_dataloader.num_workers=4 \
#         training.rollout_every=50 \
#         training.checkpoint_every=50 \
#         training.val_every=5 \
#         training.sample_every=5 \
#         task.lazy_eval=False


# python train.py --config-name='adapt_dp_moe_rgb' \
#         task=rlbench/mt4 \
#         training.num_demo=200 \
#         training.num_epochs=2001 \
#         policy.adapt_method='router+obs_encoder' \
#         model_path='output/20250227/202056_train_dp_moe_rgb_rb-mt4_N200/checkpoints/latest.ckpt' \
#         task.lazy_eval=True \
#         hydra.run.dir='output/20250228/023551_adapt_dp_moe_rgb_rb-mt4_N200' \
#         +dataloader.batch_size=256
