from diffusion_policy.policy.dp_moe import DiffusionTransformerMoePolicy as MoEPolicy
from mixture_of_experts.task_moe import TaskMoE
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torch


# @torch.no_grad()
# def augment_moe(moe: TaskMoE, num_experts: int):
#     if num_experts < moe.num_experts:
#         raise ValueError('Cannot reduce the number of experts')
#     elif num_experts == moe.num_experts:
#         return moe
    
#     # new TaskMoE
#     new_moe = TaskMoE(
#         input_size=moe.input_size,
#         head_size=moe.head_size,
#         num_experts=num_experts,
#         k=moe.k,
#         w_MI=moe.w_MI,
#         w_H=moe.w_H,
#         w_finetune_MI=moe.w_finetune_MI,
#         limit_k=moe.limit_k,
#         w_topk_loss=moe.w_topk_loss,
#         task_num=moe.task_num,
#         noisy_gating=moe.noisy_gating,
#         gating_activation=moe.gating_activation,
#         **moe.kwargs,
#     )

#     # copy the weights from the old MoE to the new MoE
#     new_moe.experts.weight[:moe.num_experts] = moe.experts.weight
#     new_moe.output_experts.weight[:moe.num_experts] = moe.output_experts.weight
#     new_moe.PTE[:, :moe.num_experts] = moe.PTE
#     new_moe.PE[:moe.num_experts] = moe.PE
#     for i, seq in enumerate(moe.f_gate):
#         new_moe.f_gate[i][-1].weight[:moe.num_experts] = seq[-1].weight
#         if seq[-1].bias is not None:
#             new_moe.f_gate[i][-1].bias[:moe.num_experts] = seq[-1].bias

#     return new_moe


# @torch.no_grad()
# def augment_policy(policy: MoEPolicy, num_experts: int):
#     moe_module_list = [
#         (name, module) 
#         for name, module in policy.named_modules() 
#         if isinstance(module, TaskMoE)
#     ]
#     for name, module in moe_module_list:
#         new_module = augment_moe(module, num_experts)
#         setattr(policy, name, new_module)
#     return policy



def main():
    policy = MoEPolicy(
        shape_meta={
            'obs': {
                'corner2_rgb': {
                    'shape': [3, 128, 128],
                    'type': 'rgb'
                },
                'behindGripper_rgb': {
                    'shape': [3, 128, 128],
                    'type': 'rgb'
                },
                'agent_pos': {
                    'shape': [9],
                    'type': 'low_dim'
                }
            },
            'action': {
                'shape': [4]
            }
        },
        noise_scheduler=DDPMScheduler(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule='squaredcos_cap_v2',
            variance_type='fixed_small',
            clip_sample=True,
            prediction_type='epsilon'
        ),
        horizon=10,
        n_tasks=1,
        n_action_steps=8,
        n_obs_steps=2,
        num_inference_steps=100,
        crop_shape=[76, 76],
        obs_encoder_group_norm=True,
        eval_fixed_crop=True,
        n_layer=1,
        n_cond_layers=0,
        n_head=4,
        n_emb=256,
        p_drop_emb=0.0,
        p_drop_attn=0.3,
        causal_attn=True,
        time_as_cond=True,
        obs_as_cond=True,
    )
    # print(policy)


    batch_size = 4
    input_obs_dict = {
        'corner2_rgb': torch.randn(batch_size, 2, 3, 128, 128),
        'behindGripper_rgb': torch.randn(batch_size, 2, 3, 128, 128),
        'agent_pos': torch.randn(batch_size, 2, 9),
    }


    for _ in range(3):
        output = policy.predict_action(input_obs_dict, task_id=0)
    action = output['action']
    print(f"before: {action.shape=}")

    
    # policy = augment_policy(policy, 10)
    policy.augment_experts(12)


    for _ in range(3):
        output = policy.predict_action(input_obs_dict, task_id=0)
    action = output['action']
    print(f"after: {action.shape=}")



if __name__ == '__main__':
    main()