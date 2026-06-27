import importlib.util
import os
import sys


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import models.pidnet as pidnet_baseline
import models.pidnet_hdp as pidnet_hdp


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_hdp_config():
    cfg_path = os.path.join(ROOT_DIR, 'configs', 'default_hdp.py')
    spec = importlib.util.spec_from_file_location('default_hdp', cfg_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cfg = module._C.clone()
    args = type('Args', (), {
        'cfg': os.path.join(
            ROOT_DIR,
            'configs',
            'power',
            'pidnet_small_local_5_dga_pappm_v1.yaml',
        ),
        'opts': [],
    })()
    module.update_config(cfg, args)
    return cfg


def main():
    cfg = load_hdp_config()

    baseline = pidnet_baseline.PIDNet(
        m=2,
        n=3,
        num_classes=cfg.DATASET.NUM_CLASSES,
        planes=32,
        ppm_planes=96,
        head_planes=128,
        augment=True,
    )

    hdp = pidnet_hdp.PIDNet(
        m=2,
        n=3,
        num_classes=cfg.DATASET.NUM_CLASSES,
        planes=32,
        ppm_planes=96,
        head_planes=128,
        augment=True,
        use_dga_pappm=True,
        dga_dir_kernel=cfg.MODEL.DGA_DIR_KERNEL,
        dga_use_gate=cfg.MODEL.DGA_USE_GATE,
    )

    baseline_params = count_params(baseline)
    hdp_params = count_params(hdp)

    print('PIDNet-S baseline Params:', baseline_params)
    print('HDP-PIDNet-DGA Params:', hdp_params)
    print('Delta Params:', hdp_params - baseline_params)
    print('PIDNet-S baseline Trainable Params:', count_trainable_params(baseline))
    print('HDP-PIDNet-DGA Trainable Params:', count_trainable_params(hdp))
    print('FLOPs: TODO')
    print('FPS: TODO')


if __name__ == '__main__':
    main()
