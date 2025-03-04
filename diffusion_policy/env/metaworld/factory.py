from diffusion_policy.env.metaworld.env import MetaworldEnv
from typing import Optional, List


MT_TASKS = {
    'mt4': [
        'door-close',
        'drawer-close',
        'disassemble',
        'window-open',
    ],
    'mt6': [
        'door-open',
        'drawer-open',
        'assembly',
        'window-close',
        'peg-insert-side',
        'hammer',
    ],
}


def is_multitask(task_name: str) -> bool:
    return task_name in MT_TASKS


def get_subtasks(task_name: str) -> List[str]:
    if is_multitask(task_name):
        return MT_TASKS[task_name]
    else:
        return [task_name]


def get_metaworld_env(
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
) -> List[MetaworldEnv]:
    
    return [
        MetaworldEnv(
            task_name=task,
            device=device,
            image_size=image_size,
            num_points=num_points,
            seed=seed,
            camera_names=camera_names,
            oracle=oracle,
            video_camera=video_camera,
            max_episode_steps=max_episode_steps
        ) for task in get_subtasks(task_name)
    ]