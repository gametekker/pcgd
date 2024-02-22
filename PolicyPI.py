import torch
import torch.nn as nn
from torch.nn import functional as F

# Policy network for agents / adversaries
# Fully-connected network with three hidden layers and ReLU activation
class PolicyPi(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()

        self.firstHidden = nn.Linear(input_dim, hidden_dim)
        self.secondHidden = nn.Linear(hidden_dim, hidden_dim)
        self.thirdHidden = nn.Linear(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, 5)

    # Forward pass
    def forward(self, s):
        outs = self.firstHidden(s)
        outs = F.relu(outs)
        outs = self.secondHidden(outs)
        outs = F.relu(outs)
        outs = self.thirdHidden(outs)
        outs = F.relu(outs)
        logits = self.classify(outs)
        return logits