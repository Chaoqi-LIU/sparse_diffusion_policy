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

# # NOTE: run this
# python train.py --config-name='train_dp_moe_rgb' \
#         task=rlbench/mt4 \
#         training.num_epochs=3501 \
#         training.num_demo=200 \
#         policy.n_layer=32 \
#         dataloader.batch_size=64 \
#         val_dataloader.batch_size=64 \
#         dataloader.num_workers=4 \
#         val_dataloader.num_workers=4 \
#         training.rollout_every=250 \
#         training.checkpoint_every=10 \
#         training.val_every=50 \
#         training.sample_every=50 \
#         task.lazy_eval=True \
#         hydra.run.dir='output/20250228/085306_train_dp_moe_rgb_rb-mt4_N200'

# python eval.py \
#         -c 'output/20250303/004123_train_dp_moe_rgb_rb-mt4_N200/checkpoints/' \
#         -o 'output/eval_moe' \
#         -n 2 \
#         --update


# python train.py --config-name='train_dp_moe_rgb' \
#         task=rlbench/mt2 \
#         training.num_epochs=2001 \
#         training.num_demo=100 \
#         policy.n_layer=8 \
#         dataloader.batch_size=256 \
#         val_dataloader.batch_size=256 \
#         dataloader.num_workers=4 \
#         val_dataloader.num_workers=4 \
#         training.rollout_every=250 \
#         training.checkpoint_every=50 \
#         training.val_every=50 \
#         training.sample_every=50 \
#         task.lazy_eval=True



# python train.py --config-name='train_dp_moe_rgb' \
#         task=metaworld/mt6 \
#         policy.n_layer=8 \
#         training.num_epochs=2001 \
#         training.num_demo=150 \
#         dataloader.batch_size=256 \
#         val_dataloader.batch_size=256 \
#         dataloader.num_workers=4 \
#         val_dataloader.num_workers=4 \
#         training.rollout_every=250 \
#         training.checkpoint_every=10 \
#         task.lazy_eval=False


# TEST DP_MOE adapt
# python train.py --config-name='adapt_dp_moe_rgb' \
#         task=metaworld/mt4 \
#         training.num_demo=100 \
#         training.num_epochs=2001 \
#         policy.adapt_method='router+obs_encoder+new_experts' \
#         model_path='output/20250304/003706_train_dp_moe_rgb_mw-mt6_N150/checkpoints/latest.ckpt' \
#         task.lazy_eval=True
# export PYTORCH_NVFUSER_DISABLE=fallback
python eval.py \
        -c 'output/20250304/214157_adapt_dp_moe_rgb_mw-mt4_N100/checkpoints/' \
        -o 'output/eval_moe' \
        -n 2 \
        --update
# unset PYTORCH_NVFUSER_DISABLE