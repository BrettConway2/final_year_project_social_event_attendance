import numpy as np
import torch


class FaceData:
    def __init__(self, embedding: torch.Tensor, prob: float, img: np.ndarray, bbox: tuple[tuple[float, float], tuple[float, float]]):
        self.embedding = embedding
        self.prob = prob
        self.img = img
        self.bbox = bbox