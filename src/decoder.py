import torch

from PreSumm.src.train_abstractive

class Decoder:
    def __init__(self, model):
        self.model = model
        pass

    def decode_model(self, src_doc, model):
        if model is None:
            model = self.model

        
        return "abcd"