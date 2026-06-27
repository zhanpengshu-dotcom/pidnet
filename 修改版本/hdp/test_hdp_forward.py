import importlib.util
import os
import sys

import torch


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import models.pidnet_hdp as pidnet_hdp


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
    config = load_hdp_config()
    model = pidnet_hdp.PIDNet(
        m=2,
        n=3,
        num_classes=config.DATASET.NUM_CLASSES,
        planes=32,
        ppm_planes=96,
        head_planes=128,
        augment=True,
        use_dga_pappm=config.MODEL.USE_DGA_PAPPM,
        dga_dir_kernel=config.MODEL.DGA_DIR_KERNEL,
        dga_use_gate=config.MODEL.DGA_USE_GATE
    )
    model.eval()

    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        y = model(x)

    assert isinstance(y, list), 'augment=True should return a list'
    assert len(y) == 3, 'PIDNet training output should have length 3'
    assert y[1].shape[1] == config.DATASET.NUM_CLASSES, 'main logits channel count mismatch'
    assert y[2].shape[1] == 1, 'boundary logits channel count should be 1'

    print('forward ok')
    print('aux_p:', tuple(y[0].shape))
    print('main:', tuple(y[1].shape))
    print('aux_d:', tuple(y[2].shape))


if __name__ == '__main__':
    main()
