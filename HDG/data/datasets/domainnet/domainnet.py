import os.path as osp
from .build_dataset import DATASET_REGISTRY
from .base_dataset import Datum, BaseDataset


@DATASET_REGISTRY.register()
class DomainNet(BaseDataset):
    """DomainNet.

    Statistics:
        - 6 distinct domains: Clipart, Infograph, Painting, Quickdraw,
        Real, Sketch.
        - Around 0.6M images.
        - 345 categories.
        - URL: http://ai.bu.edu/M3SDA/.

    Special note: the t-shirt class (327) is missing in painting_train.txt.

    Reference:
        - Peng et al. Moment Matching for Multi-Source Domain
        Adaptation. ICCV 2019.
    """
    def __init__(self, cfg):
        print("Hi")
        exit()
