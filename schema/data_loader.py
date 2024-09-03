# from schema import StrEum

from enum import Enum


class StrEnum(str, Enum):
    pass


class DataLoaderType(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
