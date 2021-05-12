from torch import nn

from .layer import MPN
from .utils import get_activation_function

class CMPNNEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = MPN(args)
    
    def forward(self, *input):
        return self.encoder(*input)


class FFN4Test(nn.Module):
    def __init__(self, args):
        super().__init__()
        first_linear_dim = args.hidden_dim
        activation = get_activation_function(args.activation)
        self.task_type = args.task_type

        if self.task_type == 'classification':
            self.sigmoid = nn.Sigmoid()
        elif self.task_type == 'multiclass':
            self.softmax = nn.Softmax(dim=2)
        else:
            raise ValueError(f'The task type ({self.task_type}) are not supported.')

        ffn = [
            nn.Linear(first_linear_dim, args.ffn_hidden_dim),
            activation,
            nn.Linear(args.ffn_hidden_dim, args.task_num)
        ]

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

    def forward(self, hidden):

        output = self.ffn(hidden)

        if self.task_type == 'classification' and not self.training:
            output = self.sigmoid(output)
        if self.task_type == 'multiclass':
            output = output.reshape((output.size(0), self.task_num, -1)) # batch size x num targets x num classes per target
            if not self.training:
                output = self.softmax(output) # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        return output
