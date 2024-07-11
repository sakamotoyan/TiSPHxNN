from enum import Enum, auto

class Mode(Enum):
    DEFAULT = auto()
    DEBUG = auto()

mode = Mode.DEFAULT