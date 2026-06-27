import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models import pidnet
from models import pidnet_hdp_v3a


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_baseline():
    return pidnet.PIDNet(
        m=2,
        n=3,
        num_classes=5,
        planes=32,
        ppm_planes=96,
        head_planes=128,
        augment=True,
    )


def build_bmhr_v3a():
    return pidnet_hdp_v3a.PIDNet(
        m=2,
        n=3,
        num_classes=5,
        planes=32,
        ppm_planes=96,
        head_planes=128,
        augment=True,
        use_bmhr=True,
        bmhr_reduction=4,
        bmhr_use_boundary_gate=True,
        bmhr_zero_init=True,
    )


def main():
    baseline = build_baseline()
    bmhr_v3a = build_bmhr_v3a()

    baseline_params = count_params(baseline)
    bmhr_params = count_params(bmhr_v3a)
    baseline_trainable = count_trainable_params(baseline)
    bmhr_trainable = count_trainable_params(bmhr_v3a)

    print('PIDNet-S baseline Params:', baseline_params)
    print('BMHR-PIDNet-v3A Params:', bmhr_params)
    print('Delta v3A - baseline Params:', bmhr_params - baseline_params)
    print('PIDNet-S baseline Trainable Params:', baseline_trainable)
    print('BMHR-PIDNet-v3A Trainable Params:', bmhr_trainable)
    print('FLOPs: TODO')
    print('FPS: TODO')


if __name__ == '__main__':
    main()
