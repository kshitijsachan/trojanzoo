#!/usr/bin/env python3

import argparse
import copy

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from trojanzoo.datasets import Dataset
from trojanzoo.utils.data import TensorListDataset
from trojanzoo.utils.logger import MetricLogger

from ...abstract import InputFiltering


def infinite_loader(loader: DataLoader):
    while True:
        for data in loader:
            yield data


def kl_div(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    p_logprobs = F.log_softmax(p_logits, dim=-1)
    q_logprobs = F.log_softmax(q_logits, dim=-1)
    return torch.sum(torch.exp(p_logprobs) * (p_logprobs - q_logprobs), dim=-1)


class Finetune(InputFiltering):
    name: str = "finetune"

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument("--ft_fpr", type=float, help="false positive rate for finetuning defense " "(default: 0.05)")
        group.add_argument("--ft_lr", type=float, help="learning rate for fine tuning defense")
        group.add_argument(
            "--ft_reg_coeff",
            type=float,
            help="regularization coefficient (weight decay for AdamW) for finetuning defense" "(default: 0.05)",
        )
        group.add_argument(
            "--ft_num_clean_samples", type=int, help="num examples for training fine tuning defense" "(default: 0.05)"
        )
        group.add_argument(
            "--ft_batch_size", type=int, help="batch size for training fine tuning defense" "(default: 0.05)"
        )
        group.add_argument("--ft_epochs", type=float, help="num of epochs for fine tuning defense" "(default: 0.05)")
        group.add_argument(
            "--ft_loss_temp",
            type=float,
            help="temperature for distillation loss for fine tuning defense" "(default: 0.05)",
        )
        # TODO: add lr scheduler
        # group.add_argument(
        #     "--ft_lr_schedule", type=str, help="false positive rate for strip defense " "(default: 0.05)"
        # )
        return group

    def __init__(
        self,
        ft_lr: float,
        ft_reg_coeff: float,
        ft_num_clean_samples: int,
        ft_batch_size: int,
        ft_epochs: int,
        ft_loss_temp: float = 2,
        ft_fpr: float = 0.05,
        **kwargs,
    ):
        assert (
            ft_num_clean_samples >= ft_batch_size
        ), f"Finetuning Defense: Number train samples ({ft_num_clean_samples}) must be larger than batch size ({ft_batch_size})."
        super().__init__(**kwargs)
        self.param_list["finetune"] = [
            "ft_lr",
            "ft_reg_coeff",
            "ft_num_clean_samples",
            "ft_batch_size",
            "ft_epochs",
            "ft_loss_temp",
            "ft_fpr",
        ]
        self.ft_fpr = ft_fpr

        # train fine tuned model
        all_clean_data = self.dataset.get_dataset(mode="train")
        ft_data = Dataset.split_dataset(all_clean_data, length=ft_num_clean_samples)[0]
        ft_loader = infinite_loader(
            self.dataset.get_dataloader(dataset=ft_data, mode="train", drop_last=True, batch_size=ft_batch_size)
        )

        ft_model = copy.deepcopy(self.model)
        for p in ft_model.parameters():
            p.requires_grad = True
        opt = torch.optim.AdamW(ft_model.parameters(), lr=ft_lr, weight_decay=ft_reg_coeff)
        losses = []
        l2_norms = []
        for epoch, batch in enumerate(ft_loader):
            if epoch == ft_epochs:
                break

            train_frac = epoch / ft_epochs
            opt.param_groups[0]["lr"] = ft_lr * (1 - train_frac)
            with torch.no_grad():
                _input, _label = self.model.get_data(batch)
                orig_logits = self.model(_input)
            ft_logits = ft_model(_input)
            loss = torch.mean(kl_div(orig_logits / ft_loss_temp, ft_logits / ft_loss_temp))
            loss.backward()
            with torch.no_grad():
                l2norm = torch.stack([torch.sum(p**2) for p in ft_model.parameters()], dim=0).sum().sqrt()
                print(f"Epoch {epoch + 1}/{ft_epochs}\tLoss: {loss.item()}\tL2 norm: {l2norm.item()}")
                losses.append(loss.item())
                l2_norms.append(l2norm.item())
            opt.step()
            opt.zero_grad()

        # plot loss and l2 norm for debugging
        if True:
            fig, ax1 = plt.subplots()

            color = "tab:red"
            ax1.set_title(f"FT Defense: train dynamics. lr={ft_lr}, reg_coeff={ft_reg_coeff}")
            ax1.set_xlabel("Epochs")
            ax1.set_ylabel("Loss", color=color)
            ax1.plot(losses, color=color)
            ax1.tick_params(axis="y", labelcolor=color)
            ax1.set_yscale("log")
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            color = "tab:blue"
            ax2.set_ylabel(
                "sum of L2 norm of all weight matrices", color=color
            )  # we already handled the x-label with ax1
            ax2.plot(l2_norms, color=color)
            ax2.tick_params(axis="y", labelcolor=color)

            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.savefig("ft_loss_l2norm.png")
            plt.close("all")
        self.ft_model = ft_model

    @torch.no_grad()
    def get_pred_labels(self) -> torch.Tensor:
        r"""Get predicted labels for test inputs.

        Returns:
            torch.Tensor: ``torch.BoolTensor`` with shape ``(2 * defense_input_num)``.
        """
        logger = MetricLogger(meter_length=40)
        str_format = "{global_avg:5.3f} ({min:5.3f}, {max:5.3f})"
        logger.create_meters(clean_score=str_format, poison_score=str_format)
        test_set = TensorListDataset(self.test_input, self.test_label)
        test_loader = self.dataset.get_dataloader(mode="valid", dataset=test_set)
        for data in logger.log_every(test_loader):
            _input, _label = self.model.get_data(data)
            trigger_input = self.attack.add_mark(_input)
            logger.meters["clean_score"].update_list(self.get_score(_input).tolist())
            logger.meters["poison_score"].update_list(self.get_score(trigger_input).tolist())
        clean_score = torch.as_tensor(logger.meters["clean_score"].deque)
        poison_score = torch.as_tensor(logger.meters["poison_score"].deque)
        clean_score_sorted = clean_score.msort()
        threshold = float(clean_score_sorted[int((1 - self.ft_fpr) * len(poison_score))])
        anomaly_score = torch.cat((clean_score, poison_score))
        print(f"Threshold: {threshold:5.3f}")
        plt.hist(anomaly_score[: len(clean_score)], bins=20, alpha=0.5, label="clean")
        plt.hist(anomaly_score[len(clean_score) :], bins=20, alpha=0.5, label="backdoor")
        plt.axvline(x=threshold, color="black", label=r"95%ile clean set threshold")
        plt.legend()
        plt.xlabel("KL Div from orig model")
        plt.title("FT Defense: Anomaly scores on test set")
        plt.savefig("ft_anomaly_scores.png")
        plt.close("all")
        return torch.where(
            anomaly_score > threshold, torch.ones_like(anomaly_score).bool(), torch.zeros_like(anomaly_score).bool()
        )

    def get_score(self, _input: torch.Tensor) -> torch.Tensor:
        orig_logits = self.model(_input)
        ft_logits = self.ft_model(_input)
        return kl_div(orig_logits, ft_logits)
