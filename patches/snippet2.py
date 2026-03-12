%%writefile src/data/compose/vision/__init__.py
from .cls.mnist import MNISTDataModule as MNIST
from .cls.cifar10 import CIFAR10DataModule as CIFAR10
from .cls.cifar100 import CIFAR100DataModule as CIFAR100
from .cls.imagenet import ImageNetDataModule as IMAGENET
from .od.voc_yolo import YOLOVOCDataModule2012 as VOC2012_YOLO
from .od.coco import COCODataModule as COCO

__all__ = [
    "MNIST",
    "CIFAR10",
    "CIFAR100",
    "IMAGENET",
    "VOC2012_YOLO",
    "COCO",
]

try:
    from .cls.cifar10_dali import CIFAR10DALIDataModule as CIFAR10_DALI
    from .cls.cifar100_dali import CIFAR10DALIDataModule as CIFAR100_DALI

    __all__ += ["CIFAR10_DALI", "CIFAR100_DALI"]
except ModuleNotFoundError:
    pass