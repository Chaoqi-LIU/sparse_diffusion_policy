import gym
import gym.spaces
import numpy as np
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
import open3d as o3d
from scipy.spatial.transform import Rotation as R
# from diffusion_policy.common.pointcloud_sampler import pointcloud_subsampling
# from diffusion_policy.common.lang_emb import get_lang_emb
from typing import Optional, Tuple, Dict, Union, List

TASK_BOUNDS = {
    'default': [
        [-0.4, 0.3, -1e-3],     # lb
        [0.4, 0.9, 0.7]         # ub
    ],
}


class MetaworldEnv(gym.Env):
    metadata = {
        "render.modes": ["rgb_array"], 
        "video.frames_per_second": 10,
    }

    def __init__(self,
        task_name: str,
        device: str = 'cuda:0',
        image_size: int = 128,
        num_points: int = 512,
        seed: Optional[int] = None,
        camera_names: List[str] = [
            'topview', 'corner', 'corner2',
            'corner3', 'behindGripper', 'gripperPOV'
        ],
        oracle: bool = False,
        video_camera: str = 'corner2',
        max_episode_steps: int = 200,
    ):
        
        # env seeding and instantiation
        self.env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[f"{task_name}-v2-goal-observable"](seed=seed)
        if seed is not None:
            self.env.seed(seed)
        self.env._freeze_rand_vec = not oracle

        # init env state
        self.env.reset()
        env_init_state = self.env.get_env_state()

        # language embedding
        # task_name_emb = get_lang_emb(task_name).cpu().numpy()

        # adjust camera near-far
        self.env.sim.model.vis.map.znear = 0.1
        self.env.sim.model.vis.map.zfar = 1.5

        # NOTE: hack corner2 camera setup, https://arxiv.org/abs/2212.05698
        cam_id = self.env.sim.model.camera_name2id('corner2')
        assert cam_id == 2      # human knowledge
        self.env.sim.model.cam_pos0[cam_id] = [0.6, 0.295, 0.8]
        self.env.sim.model.cam_pos[cam_id] = [0.6, 0.295, 0.8]

        # setup camera properties
        camera_ids = {name: self.env.sim.model.camera_name2id(name) for name in camera_names}
        camera_o3d_pinhole = {
            name: o3d.camera.PinholeCameraIntrinsic(
                image_size, image_size,
                image_size / (2 * np.tan(np.radians(self.env.sim.model.cam_fovy[cam_id]) / 2)),
                image_size / (2 * np.tan(np.radians(self.env.sim.model.cam_fovy[cam_id]) / 2)),
                image_size / 2, image_size / 2
            ) for name, cam_id in camera_ids.items()
        }
        camera_poses = {name: np.eye(4) for name in camera_names}
        for cam_name, cam_id in camera_ids.items():
            camera_poses[cam_name][:3, 3] = self.env.sim.model.cam_pos0[cam_id]
            camera_poses[cam_name][:3, :3] = np.matmul(
                self.env.sim.model.cam_mat0[cam_id].reshape(3, 3), 
                R.from_quat([1, 0, 0, 0]).as_matrix()       # mujoco cam correction
            )

        # setup gym spaces
        observation_space = gym.spaces.Dict()
        for name in camera_names:
            observation_space[f"{name}_rgb"] = gym.spaces.Box(
                low=0, high=255,
                shape=(3, image_size, image_size),
                dtype=np.uint8
            )
            # observation_space[f"{name}_depth"] = gym.spaces.Box(
            #     low=0, high=255,
            #     shape=(1, image_size, image_size),
            #     dtype=np.float32
            # )
        # observation_space["fused_pointcloud"] = gym.spaces.Box(
        #     low=-np.inf, high=np.inf,
        #     shape=(num_points, 3),
        #     dtype=np.float32
        # )
        observation_space["agent_pos"] = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=self.get_robot_state().shape,
            dtype=np.float32
        )
        # observation_space["language"] = gym.spaces.Box(
        #     low=-np.inf, high=np.inf,
        #     shape=task_name_emb.shape,
        #     dtype=np.float32
        # )
        if oracle:
            observation_space['full_state'] = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=self.env.observation_space.shape,
                dtype=np.float32
            )

        # attr
        self.env_init_state = env_init_state
        self.camera_names = camera_names
        self.camera_ids = camera_ids
        self.camera_o3d_pinhole = camera_o3d_pinhole
        self.camera_poses = camera_poses
        self.observation_space = observation_space
        self.action_space = self.env.action_space
        self.obs_sensor_dim = self.get_robot_state().shape[0]
        self.episode_length = self._max_episode_steps = max_episode_steps
        self.oracle = oracle
        self.image_size = image_size
        self.num_points = num_points
        self.task_name = task_name
        # self.task_name_emb = task_name_emb
        self.task_bbox = np.array(TASK_BOUNDS.get(task_name, TASK_BOUNDS['default']))
        self.video_camera = video_camera
        self.gpu_id = int(device.split(':')[-1])

    def get_robot_state(self) -> np.ndarray:
        return np.concatenate([
            self.env.get_endeff_pos(),
            self.env._get_site_pos('leftEndEffector'),
            self.env._get_site_pos('rightEndEffector')
        ])
    
    def get_rgb(self, cam_name: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        if cam_name is None:
            cam_name = self.camera_names
        return {
            cam: self.env.sim.render(
                width=self.image_size, height=self.image_size,
                camera_name=cam, device_id=self.gpu_id
            ) for cam in cam_name
        }
    
    # https://github.com/openai/mujoco-py/issues/520
    def get_depth(self, cam_name: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        if cam_name is None:
            cam_name = self.camera_names
        
        # depth scalin params
        extent = self.env.sim.model.stat.extent
        near = self.env.sim.model.vis.map.znear * extent
        far = self.env.sim.model.vis.map.zfar * extent

        return {
            cam: near / (1 - 
                self.env.sim.render(
                    width=self.image_size, height=self.image_size,
                    camera_name=cam, depth=True, device_id=self.gpu_id
                )[1] * (1 - near / far))
            for cam in cam_name
        }
    
    def get_pointcloud(self, 
        cam_name: Optional[List[str]] = None, 
        return_depth: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        # fuse if multiple cameras present
        if cam_name is None:
            cam_name = self.camera_names

        depths = self.get_depth(cam_name)
        result_o3d_pointcloud = o3d.geometry.PointCloud()
        for cam in cam_name:
            result_o3d_pointcloud += o3d.geometry.PointCloud.create_from_depth_image(
                o3d.geometry.Image(np.ascontiguousarray(np.flip(depths[cam], axis=0))),
                self.camera_o3d_pinhole[cam],
                np.linalg.inv(self.camera_poses[cam])
            )
        pointcloud = np.asarray(result_o3d_pointcloud.points)

        # crop and subsample
        pointcloud = pointcloud[np.all(
            (pointcloud >= self.task_bbox[0]) &
            (pointcloud <= self.task_bbox[1]),
        axis=-1)]
        pointcloud = pointcloud_subsampling(pointcloud, self.num_points, method='fps')

        if return_depth:
            return pointcloud, depths
        else:
            return pointcloud
        
    def get_obs_dict(self) -> Dict[str, np.ndarray]:
        rgbs: Dict
        depths: Dict
        pointcloud: np.ndarray
        robot_state: np.ndarray
        
        rgbs = self.get_rgb()
        # pointcloud, depths = self.get_pointcloud(return_depth=True)
        robot_state = self.get_robot_state()

        for k, v in rgbs.items():
            rgbs[k] = v.transpose(2, 0, 1)

        for k, v in depths.items():
            depths[k] = np.expand_dims(v, axis=0)

        obs_dict = {}
        for cam in self.camera_names:
            obs_dict[f"{cam}_rgb"] = rgbs[cam]
            # obs_dict[f"{cam}_depth"] = depths[cam]
        # obs_dict["fused_pointcloud"] = pointcloud
        obs_dict["agent_pos"] = robot_state
        # obs_dict["language"] = self.task_name_emb
        return obs_dict
    
    def step(self, action: np.ndarray):
        full_state, reward, done, info = self.env.step(action)
        self.cur_step += 1
        obs_dict = self.get_obs_dict()
        if self.oracle:
            obs_dict['full_state'] = full_state
        done = done or self.cur_step >= self.episode_length
        return obs_dict, reward, done, info
    
    def reset(self) -> Dict[str, np.ndarray]:
        self.env.reset()
        self.env.reset_model()
        full_state = self.env.reset()
        self.env.set_env_state(self.env_init_state)
        self.cur_step = 0
        obs_dict = self.get_obs_dict()
        if self.oracle:
            obs_dict['full_state'] = full_state
        return obs_dict

    def seed(self, seed=None):
        self.env.seed(seed)
        self.init_env_state = self._env_init_state_of_seed(seed)

    def render(self, mode='rgb_array'):
        # NOTE: only for video recording wrapper
        assert mode == 'rgb_array'
        return self.get_rgb([self.video_camera])[self.video_camera]
    
    def close(self):
        self.env.close()
    
    def _env_init_state_of_seed(self, seed: int):
        env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[f"{self.task_name}-v2-goal-observable"](seed=seed)
        env.reset()
        init_state = env.get_env_state()
        env.close()
        return init_state
    