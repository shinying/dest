import torch
from torchmetrics import Metric


class Accuracy(Metric):
    def __init__(self, dist_sync_on_step=False, topk=1):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.topk = topk
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits = logits.detach().to(self.correct.device)
        target = target.detach().to(self.correct.device)

        logits = logits[target!=-100]
        target = target[target!=-100]
        if target.numel() == 0:
            return 1

        if self.topk == 1:
            preds = logits.argmax(dim=-1)
            acc = torch.sum(preds==target)
        else:
            preds = logits.topk(k=self.topk)[1]
            acc = (preds==target.unsqueeze(1)).any(dim=1).sum()

        self.correct += acc
        self.total += target.numel()

    def compute(self):
        return self.correct / self.total


class Scalar(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)
        self.scalar += scalar
        self.total += 1

    def compute(self):
        return self.scalar / self.total
