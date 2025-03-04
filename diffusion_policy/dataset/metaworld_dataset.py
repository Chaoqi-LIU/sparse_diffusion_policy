import torch
import numpy as np
import copy
import os
from PIL import Image
from matplotlib import pyplot as plt
import open3d as o3d
import time
import tqdm

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset

from typing import Dict, Optional, List



class MetaworldDataset(BaseImageDataset):
    def __init__(
        self,
        zarr_path: str,
        obs_keys: List[str],
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0, 
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: Optional[int] = None,
    ):
        super().__init__()
        print(f"{zarr_path=}")
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['action', *obs_keys],
        )
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed,
        )
        self.seq_sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask
        )

        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.obs_keys = obs_keys


    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.seq_sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set
    

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            **{k: self.replay_buffer[k] for k in self.obs_keys}
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer
    

    def __len__(self):
        return len(self.seq_sampler)
    

    def _sample_to_data(self, sample):
        data = {
            'obs': {k: sample[k].astype(np.float32) for k in self.obs_keys},
            'action': sample['action'].astype(np.float32),
        }
        return data
    

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.seq_sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, lambda x: torch.from_numpy(x))
        return torch_data


    def replay(self, obs_key: str, dt: float = 0.1):
        try:
            if obs_key.endswith('pointcloud'):
                o3dvis = o3d.visualization.Visualizer()
                o3dvis.create_window()
                pcd = None

            for idx in tqdm.tqdm(range(len(self)), desc=f'MetaworldDataset replay {obs_key}'):
                sample = self.seq_sampler.sample_sequence(idx)
                obs = self._sample_to_data(sample)['obs'][obs_key][0]
                if obs_key.endswith('rgb'):
                    obs = np.moveaxis(obs, 0, -1)
                    Image.fromarray(obs.astype(np.uint8)).save('rgb.png')
                    time.sleep(dt)
                elif obs_key.endswith('depth'):
                    obs = (obs - obs.min()) / (obs.max() - obs.min())
                    plt.imsave('depth.png', obs)
                    time.sleep(dt)
                elif obs_key.endswith('pointcloud'):
                    if pcd is None:
                        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obs))
                        o3dvis.add_geometry(pcd)
                    else:
                        pcd.points = o3d.utility.Vector3dVector(obs)
                        o3dvis.update_geometry(pcd)
                    o3dvis.poll_events()
                    o3dvis.update_renderer()
                    time.sleep(dt)
        except KeyboardInterrupt as e:
            print(f"An error occurred: {e}")
            if os.path.exists('rgb.png'):
                os.remove('rgb.png')
            if os.path.exists('depth.png'):
                os.remove('depth.png')