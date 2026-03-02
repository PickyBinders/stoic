import torch
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics import Metric


def log_confusion_matrix_advanced(
    conf_matrix,
    class_names=None,
    step=None,
    normalize=False,
    title="Confusion Matrix",
    split="train",
    cmap=sns.color_palette("YlOrRd", as_cmap=True),
    figure_size=(10, 8),
):
    """
    Advanced version with more customization options

    Args:
        conf_matrix (torch.Tensor): Confusion matrix from torchmetrics
        class_names (list): List of class names
        step (int): Current step/epoch for logging
        normalize (bool): Whether to normalize the confusion matrix
        title (str): Plot title
        cmap (str): Color map for the plot
        figure_size (tuple): Figure size in inches
    """
    if isinstance(conf_matrix, torch.Tensor):
        conf_matrix = conf_matrix.cpu().numpy()

    fmt = ".2f"
    if class_names is None:
        class_names = [f"Class {i}" for i in range(conf_matrix.shape[0])]

    plt.figure(figsize=figure_size)
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0,
        vmax=1,
    )

    plt.title(f"{title} - {split}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    table_data = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            table_data.append(
                [
                    class_names[i],
                    class_names[j],
                    int(conf_matrix[i, j]),
                    float(
                        conf_matrix[i, j] / conf_matrix[i].sum()
                    ),
                ]
            )

    wandb.log(
        {
            f"confusion_matrix/plot_{split}": wandb.Image(plt),
            f"confusion_matrix/data_{split}": wandb.Table(
                data=table_data, columns=["True", "Predicted", "Count", "Percentage"]
            ),
        },
        step=step,
    )

    plt.close()


class WeightDistributionMetric(Metric):
    def __init__(self, percentages=[0.1, 0.25, 0.5], dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.percentages = percentages

        self.add_state("all_weights", default=[], dist_reduce_fx=None)
        self.add_state("all_masks", default=[], dist_reduce_fx=None)

    def update(self, attention_weights: torch.Tensor, mask: torch.Tensor = None):
        """
        Accumulate weights and masks for later computation

        Args:
            attention_weights: tensor of shape (batch_size, seq_length)
            mask: boolean tensor of shape (batch_size, seq_length) where True means ignore
        """
        for b in range(attention_weights.shape[0]):
            self.all_weights.append(attention_weights[b].detach())
            if mask is not None:
                self.all_masks.append(mask[b].detach())
            else:
                self.all_masks.append(None)

    def compute(self):
        """Compute statistics on all accumulated weights at epoch end"""
        results = {}

        for p in self.percentages:
            all_top_k_sums = []

            for weights, mask in zip(self.all_weights, self.all_masks):
                if mask is not None:
                    valid_weights = weights[~mask]
                    seq_length = (~mask).sum()
                else:
                    valid_weights = weights
                    seq_length = len(weights)

                sorted_weights, _ = torch.sort(valid_weights, descending=True)
                k = max(1, int(seq_length * p))
                top_k_sum = sorted_weights[:k].sum()
                all_top_k_sums.append(top_k_sum)

            mean = torch.stack(all_top_k_sums).mean()
            results[f"top_{int(p*100)}p_weight_sum"] = mean

        return results

    def reset(self):
        """Clear accumulated weights and masks"""
        self.all_weights = []
        self.all_masks = []


class WeightOverlapMetric(Metric):
    """
    TorchMetrics implementation for measuring overlap between predicted
    residue contact probabilities and true contacting residues.

    This metric computes precision, recall, and F1-score between
    the predicted residue contact probabilities and the true interacting residues.
    """

    def __init__(self, threshold=0.5, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.threshold = threshold

        self.add_state(
            "total_precision", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("total_recall", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_f1", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predicted_probs, target_weights):
        """
        Update the metric state with new predictions and targets.

        Args:
            predicted_probs: Tensor of shape [batch_size, seq_len] with predicted contact probabilities
            target_weights: Tensor of shape [batch_size, seq_len] or [batch_size, seq_len, 1] with binary labels
        """
        if target_weights.dim() == 3:
            target_weights = target_weights.squeeze(2)

        target_weights = target_weights.to(predicted_probs.device)

        predicted_binary = (predicted_probs >= self.threshold).float()

        target_mask = target_weights > 0
        num_target = target_mask.sum(dim=1)

        valid_mask = num_target > 0
        valid_count = valid_mask.sum()

        if valid_count == 0:
            return

        true_positives = (predicted_binary * target_mask).sum(dim=1)
        false_positives = (predicted_binary * (~target_mask)).sum(dim=1)
        false_negatives = ((1 - predicted_binary) * target_mask).sum(dim=1)

        precision = torch.zeros_like(true_positives)
        non_zero_pred = (true_positives + false_positives) > 0
        precision[non_zero_pred] = true_positives[non_zero_pred] / (
            true_positives[non_zero_pred] + false_positives[non_zero_pred]
        )

        recall = torch.zeros_like(true_positives)
        non_zero_actual = num_target > 0
        recall[non_zero_actual] = (
            true_positives[non_zero_actual] / num_target[non_zero_actual]
        )

        f1 = torch.zeros_like(precision)
        non_zero_f1 = (precision + recall) > 0
        f1[non_zero_f1] = (
            2
            * (precision[non_zero_f1] * recall[non_zero_f1])
            / (precision[non_zero_f1] + recall[non_zero_f1])
        )

        self.total_precision += precision[valid_mask].sum()
        self.total_recall += recall[valid_mask].sum()
        self.total_f1 += f1[valid_mask].sum()
        self.total_count += valid_count

    def compute(self):
        """
        Compute the final metrics from accumulated statistics.

        Returns:
            Dict containing the average precision, recall, and F1-score values
        """
        if self.total_count == 0:
            return {
                "precision": torch.tensor(0.0),
                "recall": torch.tensor(0.0),
                "f1": torch.tensor(0.0),
            }

        avg_precision = self.total_precision / self.total_count
        avg_recall = self.total_recall / self.total_count
        avg_f1 = self.total_f1 / self.total_count

        return {"precision": avg_precision, "recall": avg_recall, "f1": avg_f1}
