if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import omegaconf
from omegaconf import OmegaConf, open_dict
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import dill
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.dp_moe import DiffusionTransformerMoePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from typing import List
from diffusion_policy.dataset.multitask_dataset import MultiDataLoader
from itertools import zip_longest
OmegaConf.register_new_resolver("eval", eval, replace=True)

class AdaptMoePolicyWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        payload = torch.load(open(cfg.model_path, 'rb'), pickle_module=dill)
        pretrain_cfg: OmegaConf = payload['cfg']

        # load pretrained model
        cls = hydra.utils.get_class(pretrain_cfg._target_)
        workspace: BaseWorkspace = cls(pretrain_cfg, output_dir=output_dir)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        self.model: DiffusionTransformerMoePolicy = workspace.model
        try:
            if cfg.training.use_ema:
                self.model = workspace.ema_model
        except omegaconf.errors.ConfigAttributeError:
            pass

        # update with adapt config
        with open_dict(pretrain_cfg):
            cfg = OmegaConf.merge(pretrain_cfg, cfg)

        # introduce new unets if needed
        if 'new_experts' in cfg.policy.adapt_method:
            new_experts = 2
            print(f"Augmenting experts by {new_experts}")
            self.model.augment_experts(num_experts=self.model.num_experts + new_experts)  # TODO: make this configurable
        assert len(self.model.normalizers) == 1, len(self.model.normalizers)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.ema_model: DiffusionTransformerMoePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        # configure training state
        self.global_step = 0
        self.epoch = 0

        # update workspace config
        self.cfg = cfg
        self.adapt_method = cfg.policy.adapt_method

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        assert cfg.task_num == 1, "Currently we only allow task_num=1"

        # resume training
        if cfg.training.resume:   
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
                print(f"{self.epoch=}, {self.global_step=}")
                if self.epoch >= cfg.training.num_epochs:
                    print("Training already completed")
                    return
                
        # report status
        self.model.report_model_status()

        # configure dataset
        datasets: List[BaseImageDataset] = []
        for i in range(cfg.task_num):
            datasets.append(hydra.utils.instantiate(cfg[f'task{i}'].dataset))
        
        assert isinstance(datasets[0], BaseImageDataset)
        train_dataloaders = []
        normalizers=[]
        for dataset in datasets:
            train_dataloaders.append(DataLoader(dataset, **cfg.dataloader))
            normalizers.append(dataset.get_normalizer())
        assert len(train_dataloaders) == cfg.task_num
        max_train_dataloader_len = max([len(train_dataloader) for train_dataloader in train_dataloaders])
        for train_dataloader in train_dataloaders:
            print("Length of train_dataloader: ", len(train_dataloader))
        print(f"max_train_dataloader_len: {max_train_dataloader_len}")
        multi_traindataloader=MultiDataLoader(train_dataloaders)
        # multi_traindataloader.get_memory_usage()
        # configure validation dataset
        val_datasets=[]
        for dataset in datasets:
            val_datasets.append(dataset.get_validation_dataset())
       
        val_dataloaders = []
        for val_dataset in val_datasets:
            val_dataloaders.append(DataLoader(val_dataset, **cfg.val_dataloader))

        self.model.set_normalizer(normalizers)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizers)
        assert len(self.model.normalizers) == 1, len(self.model.normalizers)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                max_train_dataloader_len * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        lazy_eval = cfg.task.lazy_eval
        if not lazy_eval:
            env_runners = []
            # env_runner3: BaseImageRunner
            for i in range(cfg.task_num):
                env_runners.append(hydra.utils.instantiate(cfg[f'task{i}'].env_runner, output_dir=self.output_dir))
                assert isinstance(env_runners[i], BaseImageRunner), type(env_runners[i])

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        for normalizer in self.model.normalizers:
            normalizer.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
            for normalizer in self.ema_model.normalizers:
                normalizer.to(device)
        optimizer_to(self.optimizer, device)
        self.model.adapt(method=self.adapt_method)
        
        # save batch for sampling
        train_sampling_batchs = []
        for i in range(cfg.task_num):
            train_sampling_batchs.append(None)
        test_sampling_batchs = []
        for i in range(cfg.task_num):
            test_sampling_batchs.append(None)

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        num_total_params = sum(p.numel() for p in self.model.parameters())
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            while self.epoch < cfg.training.num_epochs:
                step_log = dict()
                # ========= train for this epoch ==========

                num_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

                train_losses = list()
                with tqdm.tqdm(multi_traindataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx,batch in enumerate(tepoch):

                        assigned_task_id = batch_idx%cfg.task_num
                        

                        # load the next batch of the dataloader, 'DataLoader' object is not an iterator
                        assert assigned_task_id == multi_traindataloader.loader_idx
                        if batch is None:
                            continue
                        if train_sampling_batchs[assigned_task_id] is None:
                            train_sampling_batchs[assigned_task_id] = batch
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        task_id = torch.tensor([assigned_task_id], dtype=torch.int64).to(device)
                

                        # compute loss
                        raw_loss = self.model.compute_loss(batch,task_id)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # wipe out the gradients of old experts if applicable
                        num_zeroed_out_grad = 0
                        if 'new_experts' in self.adapt_method:
                            num_zeroed_out_grad = self.model.zero_out_old_experts_grad()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0],
                            'trainable_param_ratio': (num_trainable_params - num_zeroed_out_grad) / num_total_params
                        }

                        is_last_batch = (batch_idx == (max_train_dataloader_len-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                for i, train_sampling_batch in enumerate(train_sampling_batchs):
                    if train_sampling_batch is None:
                        raise ValueError(f"train_sampling_batch {i} is None")
                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                runner_logs = []
                if (self.epoch % cfg.training.rollout_every) == 0:
                    if not lazy_eval:
                        for i, env_runner in enumerate(env_runners):
                            runner_log = env_runner.run(policy,task_id=torch.tensor([i], dtype=torch.int64).to(device))
                            # runner_log = {key + f'_{i}': value for key, value in runner_log.items()}
                            runner_logs.append(runner_log)
                        for runner_log in runner_logs:
                            step_log.update(runner_log)
                    else:
                        step_log[topk_manager.monitor_key] = self.epoch / cfg.training.num_epochs
                        
                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses_list = []
                        for i in range(cfg.task_num):
                            val_losses_list.append([])
                        zip_val_dataloaders = zip_longest(*val_dataloaders)
                        # val_losses3 = list()
                        with tqdm.tqdm(zip_val_dataloaders, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batches in enumerate(tepoch):
                                assigned_task_id = batch_idx%cfg.task_num
                                for i, batch in enumerate(batches):
                                    if batch is None:
                                        continue
                                    if test_sampling_batchs[assigned_task_id] is None:
                                        test_sampling_batchs[assigned_task_id] = batch
                                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                    loss = self.model.compute_loss(batch,task_id=torch.tensor([i], dtype=torch.int64).to(device))
                                    val_losses_list[i].append(loss)
                                    if (cfg.training.max_val_steps is not None) \
                                        and batch_idx >= (cfg.training.max_val_steps-1):
                                        break
                        if len(val_losses_list[0]) > 0:
                            for i, val_losses in enumerate(val_losses_list):
                                val_loss = torch.mean(torch.tensor(val_losses)).item()
                                # step_log[f'val_loss_{i}'] = val_loss
                                step_log['val_loss'] = val_loss
                            # step_log['val_loss3'] = val_loss3
                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        for tag, sample_batches in [
                            ('train_action_mse', train_sampling_batchs),
                            ('test_action_mse', test_sampling_batchs)
                        ]:
                            for i, sample_batch in enumerate(sample_batches):
                                assert sample_batch is not None
                                batch = dict_apply(sample_batch, lambda x: x.to(device, non_blocking=True))
                                obs_dict = batch['obs']
                                gt_action = batch['action']
                                result = policy.predict_action(obs_dict,task_id=torch.tensor([i], dtype=torch.int64).to(device))
                                pred_action = result['action_pred']
                                mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                                # step_log[f'{tag}_{i}'] = mse.item()
                                step_log[tag] = mse.item()
                            del batch
                            del obs_dict
                            del gt_action
                            del result
                            del pred_action
                            del mse
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
     
                    # sum=0
                    # for key in metric_dict.keys():
                    #     # if start with cfg.checkpoint.topk.monitor_key, then sum up
                    #     if key.startswith(cfg.checkpoint.topk.monitor_key):
                    #         sum+=metric_dict[key]
                    # metric_dict[cfg.checkpoint.topk.monitor_key] = sum
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                        if lazy_eval:
                            step_log.pop(topk_manager.monitor_key)

                # ========= eval end for this epoch ==========
                policy.adapt(method=self.adapt_method)

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1
                multi_traindataloader.reset()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = AdaptMoePolicyWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
