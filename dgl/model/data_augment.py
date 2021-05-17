from torch import nn
class DropoutGenerator(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, in_feature):
        return self.dropout1(in_feature), self.dropout2(in_feature)