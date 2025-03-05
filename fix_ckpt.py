import os
import dill
import torch
import hydra
from diffusion_policy.workspace.base_workspace import BaseWorkspace

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)


dir = 'output/20250304/003706_train_dp_moe_rgb_mw-mt6_N150/checkpoints/'
ckpts = [
    os.path.join(dir, f)
    for f in os.listdir(dir)
    if f.endswith('.ckpt') and f != 'latest.ckpt'
]

for ckpt in ckpts:
    payload = torch.load(open(ckpt, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    assert cfg.task_num == 1, "Only allow task_num = 1"
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir='output/')
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    # assert len(policy.normalizers) == 0, len(policy.normalizers)
    print(f"{len(policy.normalizers)=}")

    # if len(policy.normalizers) == 0:
    #     dataset = hydra.utils.instantiate(cfg.task0.dataset)
    #     policy.set_normalizer([dataset.get_normalizer()])
    #     policy.normalizers = torch.nn.ModuleList(policy.normalizers)

    # # save
    # workspace.save_checkpoint(path=ckpt)