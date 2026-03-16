import torch
import torch.nn as nn
from pysot.models.model_builder import ModelBuilder

class InferenceWrapper(nn.Module):
    def __init__(self, model):
        super(InferenceWrapper, self).__init__()
        self.model = model

    def forward(self, z, x):
        self.model.template(z)
        out = self.model.track(x)
        return out['cls'], out['loc']

