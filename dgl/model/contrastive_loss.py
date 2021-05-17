import torch
from torch import nn
import torch.nn.functional as F

def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

class DropoutPNGenerator(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout_anchor = nn.Dropout(dropout)
        self.dropout_positive = nn.Dropout(dropout)

    def forward(self, emb):
        # z_i = F.normalize(self.dropout_anchor(emb), dim=1)
        # z_j = F.normalize(self.dropout_positive(emb), dim=1)
        # z_k = F.normalize(emb, dim=1)
        z_i = self.dropout_anchor(emb)
        z_j = self.dropout_positive(emb)
        z_k = emb
        # z_i = self.dropout_anchor(z_k)
        # z_j = self.dropout_positive(z_k)

        return z_i, z_j, z_k

class AveragePNGenerator(nn.Module):
    def __init__(self, ratio=0.1):
        super().__init__()
        self.ratio = ratio

    def forward(self, emb):
        z_i = F.normalize(emb, dim=1)
        z_j = F.normalize(self.ratio*torch.sum(emb, dim=0) + (1-2*self.ratio)*emb, dim=1)

        return z_i, z_j, z_i

class LinearPNGenerator(nn.Module):
    def __init__(self, in_feature_size):
        super().__init__()
        self.linear1 = nn.Linear(in_feature_size, in_feature_size)
        self.linear2 = nn.Linear(in_feature_size, in_feature_size)

    def forward(self, emb):
        z_i = F.normalize(self.linear1(emb), dim=1)
        z_j = F.normalize(self.linear2(emb), dim=1)
        z_k = F.normalize(emb, dim=1)

        return z_i, z_j, z_k

class HalfLinearPNGenerator(nn.Module):
    def __init__(self, in_feature_size):
        super().__init__()
        self.linear1 = nn.Linear(in_feature_size, in_feature_size//2)
        self.linear2 = nn.Linear(in_feature_size, in_feature_size//2)
        self.linear3 = nn.Linear(in_feature_size, in_feature_size//2)

    def forward(self, emb):
        z_i = F.normalize(self.linear1(emb), dim=1)
        z_j = F.normalize(self.linear2(emb), dim=1)
        z_k = F.normalize(self.linear3(emb), dim=1)

        return z_i, z_j, z_k

class SquareSimilarity(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
    
    def forward(self, anchor, positive, negative, temperature=0.1):
        batch_size = anchor.size(0)

        positive_similarity = F.cosine_similarity(anchor, positive, dim=1).view(-1, 1)
        negative_mask = (~torch.eye(batch_size, batch_size, dtype=bool)).to(self.device).float()
        similarity = F.cosine_similarity(torch.cat([anchor], dim=0).unsqueeze(1), torch.cat([negative], dim=0).unsqueeze(0), dim=2) * negative_mask
        first_column = similarity[:, 0]
        similarity = torch.diag(first_column) + similarity

        similarity = torch.cat((positive_similarity, similarity[:, 1:]), dim=1)
        return similarity / temperature

class SoftmaxLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, similarity):
        softmax_similarity = F.log_softmax(similarity, dim=1)
        loss = -torch.sum(softmax_similarity[:, 0])
        return loss

class NCESoftmaxLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.device = device

    def forward(self, similarity):
        batch_size = similarity.size(0)
        label = torch.tensor([0]*batch_size).to(self.device).long()
        loss = self.criterion(similarity, label)
        return loss


PN_GENERATOR = {
    'Dropout': DropoutPNGenerator,
    'Average': AveragePNGenerator,
    'Linear': LinearPNGenerator,
    'HalfLinear': HalfLinearPNGenerator,
}

CONTRASTIVE_LOSS = {
    'Softmax': SoftmaxLoss,
    'NCESoftmax': NCESoftmaxLoss,
}

class ContrastiveLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        if args.pn_generator == 'Dropout' or args.pn_generator == 'Average':
            self.pn_generator = PN_GENERATOR[args.pn_generator]()
        elif args.pn_generator == 'Linear' or args.pn_generator == 'HalfLinear':
            self.pn_generator = PN_GENERATOR[args.pn_generator](args.hidden_dim)
        else:
            raise NotImplementedError(f'Not Supportted Dataset {args.pn_generator}')
        
        self.similarity = SquareSimilarity(self.device)

        if args.contrastive_loss in CONTRASTIVE_LOSS:
            self.contrastive_loss = CONTRASTIVE_LOSS[args.contrastive_loss](self.device)
        else:
            raise NotImplementedError(f'Not Supportted Dataset {args.contrastive_loss}')

    def forward(self, emb):
        anchor, positive, negative = self.pn_generator(emb)
        similarity = self.similarity(anchor, positive, negative)
        loss = self.contrastive_loss(similarity)

        return loss

class NCESoftmaxContrastiveLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.device = args.device
        self.temperature = args.temperature
    
    def forward(self, in_feature_i, in_feature_j):
        batch_size = in_feature_i.size(0)

        positive_similarity = F.cosine_similarity(in_feature_i, in_feature_j, dim=1).view(-1, 1)
        negative_mask = (~torch.eye(batch_size, batch_size, dtype=bool)).to(self.device).float()
        similarity = F.cosine_similarity(torch.cat([in_feature_i], dim=0).unsqueeze(1), torch.cat([in_feature_j], dim=0).unsqueeze(0), dim=2) * negative_mask
        first_column = similarity[:, 0]
        similarity = torch.diag(first_column) + similarity

        similarity = torch.cat((positive_similarity, similarity[:, 1:]), dim=1) / self.temperature
        label = torch.tensor([0]*batch_size).to(self.device).long()
        loss = self.criterion(similarity, label)
        return loss
            
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    
    similarity = SquareSimilarity(args.device)
    print(similarity(torch.tensor([[1.0, 2.0], [3.0, -2.0], [-5, 1]]), torch.tensor([[1.0, 2.0], [3.0, -2.0], [-5, 1]]), torch.tensor([[1.0, 2.0], [3.0, -2.0], [-5, 1]])))