"""
Usage:
python experiments/eval_policy_sim.py --checkpoint path/to/ckpt -o path/to/output_dir
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import re
import pathlib
import click
import hydra
import omegaconf
import torch
import dill
import wandb
import json
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from typing import List

@click.command()
@click.option('-c', '--checkpoint', required=True, help="either a .ckpt file or a directory containing .ckpt files")
@click.option('-o', '--output_dir', required=True, help="output directory for eval info dump")
@click.option('-n', '--num_exp', default=3, help="num experiments to run")
@click.option('-d', '--device', default='cuda:0', help="device to run on")
@click.option('-u', '--update', is_flag=True, help="weather to update `success_rate` in ckpt file name")
@click.option('-s', '--dropout_obs', default=(), multiple=True, help="a list of sensor ports to drop i.e. set to 0s")
def eval_policy_sim(checkpoint, output_dir, num_exp, device, update, dropout_obs):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # grab all checkpoints
    ckpts: List[str]    # file paths to checkpoints to evaluate
    if os.path.isdir(checkpoint):
        ckpts = [
            os.path.join(checkpoint, f) 
            for f in os.listdir(checkpoint) 
            if f.endswith('.ckpt') and f != 'latest.ckpt'
        ]
    else:
        ckpts = [checkpoint,]

    base_output_dir = output_dir
    for ckpt in ckpts:
        # format output dir
        if len(ckpts) > 1:
            output_dir = os.path.join(base_output_dir, os.path.basename(ckpt).replace('.ckpt', ''))
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        else:
            output_dir = base_output_dir
        
        # load checkpoint
        payload = torch.load(open(ckpt, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        assert cfg.task_num == 1, "Only allow task_num = 1"
        cls = hydra.utils.get_class(cfg._target_)
        workspace: BaseWorkspace = cls(cfg, output_dir=output_dir)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        
        # get policy from workspace
        policy: BaseImagePolicy = workspace.model
        try:
            if cfg.training.use_ema:
                policy = workspace.ema_model
        except omegaconf.errors.ConfigAttributeError:
            # compositional policy does not have ema_model
            pass

        device = torch.device(device)

        # load normalizer, this is a historical bug in sdp ...
        if len(policy.normalizers) == 0:
            # cfg.task0.dataset.zarr_path = 'data/rlbench/mt4_expert_200.zarr'
            dataset = hydra.utils.instantiate(cfg.task0.dataset)
            policy.set_normalizer([dataset.get_normalizer().to(device)])
            del dataset
        
        policy.to(device)
        policy.eval()

        # dropout obs ports
        dropout_obs_str = ','.join(dropout_obs)
        print(f"Dropout obs ports: {dropout_obs_str}")
        
        # run eval
        print(f"Running evaluation on {ckpt}")
        env_runner: BaseImageRunner = hydra.utils.instantiate(
            cfg.task0.env_runner,
            output_dir=output_dir,
        )
        
        runner_log = env_runner.run(policy, task_id=torch.tensor([0], dtype=torch.int64).to(device))
        for key, value in runner_log.items():
            if isinstance(value, wandb.sdk.data_types.video.Video):
                runner_log[key] = [value]
        print(f"Exp 1: success rate = {runner_log['mean_success_rate']}")
        for i in range(num_exp - 1):
            this_log = env_runner.run(policy, task_id=torch.tensor([0], dtype=torch.int64).to(device))
            print(f"Exp {i + 2}: success rate = {this_log['mean_success_rate']}")
            # merge logs
            for key, value in this_log.items():
                assert key in runner_log
                if isinstance(value, wandb.sdk.data_types.video.Video):
                    runner_log[key].append(value)
                else:
                    runner_log[key] += value
        # take average
        for key, value in runner_log.items():
            if not isinstance(value, list):
                runner_log[key] = value / num_exp
        env_runner.close()
        
        # dump log to json
        json_log = dict()
        json_log['checkpoint'] = ckpt
        json_log['num_exp'] = num_exp
        json_log['dropout_obs'] = dropout_obs_str
        for key, value in runner_log.items():
            if isinstance(value, list):
                for i, video in enumerate(value):
                    assert isinstance(video, wandb.sdk.data_types.video.Video)
                    json_log[f'{key}_{i}'] = video._path
            else:
                json_log[key] = value
        out_path = os.path.join(output_dir, 'eval_log.json')
        json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

        # update the checkpoint name
        if update:
            new_ckpt_path = os.path.join(
                os.path.dirname(ckpt), 
                re.sub(
                    r'\b\d\.\d{3}\.ckpt$',
                    f"{runner_log['mean_success_rate']:.3f}.ckpt", 
                    os.path.basename(ckpt)
                )
            )
            os.rename(ckpt, new_ckpt_path)
            print(f"{ckpt} -> {new_ckpt_path}")


if __name__ == '__main__':
    eval_policy_sim()