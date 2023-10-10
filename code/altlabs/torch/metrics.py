import torch


def top_k_accuracy(pred_probas: torch.Tensor, target: torch.Tensor, k: int) -> torch.Tensor:
    batch_size = target.size(0)

    _, pred = pred_probas.topk(k, 1, True, True)
    pred = pred.t()
    correct, _ = pred.eq(target.view(1, -1).expand_as(pred)).max(0)

    return correct.float().sum() / batch_size


def binary_accuracy(pred_probas: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    y_pred_tag = torch.round(pred_probas)

    correct_results_sum = y_pred_tag.eq(target).sum().float()
    return correct_results_sum / target.numel()
