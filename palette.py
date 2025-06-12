import numpy as np


class Palette():
    def __init__(self, colours: np.ndarray, counts: np.ndarray):
        assert(len(colours) == len(counts))
        self.colours = colours.astype(np.float64)
        self.counts = counts
        self.size = len(counts)