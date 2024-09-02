from enum import Enum


class EpochType(Enum):
    TRAIN = 'train'
    EVAL = 'val'
    def __repr__(self):
        # Возвращаем только значение перечисления
        return self.value