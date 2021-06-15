import numpy
import pandas

import torch

class AbstractiveModel:
    def __init__(self):
        self.encoder = 0
        self.decoder = 0

    def train(self, data):
        return ""