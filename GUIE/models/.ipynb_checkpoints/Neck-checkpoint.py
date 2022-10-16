import torch
import torch.nn as nn
import torch.nn.functional as F

class Neck(nn.Module):
    def __init__(self, in_features, out_features, style='high_dim'):
        super().__init__()
        if style=='simple':
            self.neck = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features, out_features),
            )
        elif style=='norm_double_linear':
            self.neck = nn.Sequential(
                nn.BatchNorm1d(in_features),
                nn.Dropout(0.3),
                nn.Linear(in_features, out_features*2),
                nn.Linear(out_features*2, out_features),
            )
        elif stype=='norm_single_linear':
            self.neck = nn.Sequential(
                nn.BatchNorm1d(in_features),
                nn.Dropout(0.2),
                nn.Linear(in_features, out_features),
            )
        else:
            raise NotImplementedError(f"Unkown Neck: {stype}")

    def forward(self, x):
        return self.neck(x) 