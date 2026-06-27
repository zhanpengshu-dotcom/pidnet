import os
import sys

import torch


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models import pidnet_hdp_v3a


def main():
    model = pidnet_hdp_v3a.PIDNet(
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
    model.eval()

    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        y = model(x)

    assert isinstance(y, list), type(y)
    assert len(y) == 3, len(y)
    assert tuple(y[1].shape) == (1, 5, 64, 64), tuple(y[1].shape)
    assert y[2].shape[1] == 1, tuple(y[2].shape)

    print('forward ok')
    print('main:', tuple(y[1].shape))


if __name__ == '__main__':
    main()
