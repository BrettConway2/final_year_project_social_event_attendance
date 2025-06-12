
import numpy as np

from palette import Palette


class Feature:
    def __init__(self, image, colour: np.ndarray, centre: tuple[float, float], mask: np.ndarray, colour_palette: Palette, w: float, h: float, info="", embedding = None):
        self.colour = colour
        self.centre = centre
        self.mask = mask
        self.colour_palette = colour_palette
        self.h = h
        self.w = w
        self.info = info
        self.image = image
        self.embedding = embedding
