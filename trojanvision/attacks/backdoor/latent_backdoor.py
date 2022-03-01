#!/usr/bin/env python3

from ..abstract import BackdoorAttack

from trojanvision.environ import env
from trojanzoo.utils.data import sample_batch
from trojanzoo.utils.tensor import to_tensor, tanh_func

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import functools

from typing import TYPE_CHECKING
import argparse
from collections.abc import Callable
if TYPE_CHECKING:
    import torch.utils.data


class LatentBackdoor(BackdoorAttack):
    r"""
    | Latent Backdoor proposed by Yuanshun Yao, Huiying Li, Haitao Zheng
      and Ben Y. Zhao from University of Chicago in CCS 2019.
    | It inherits :class:`trojanvision.attacks.BackdoorAttack`.
    |
    | Based on :class:`trojanvision.attacks.BadNet`
      and similar to :class:`trojanvision.attacks.TrojanNN`,
      Latent Backdoor preprocesses watermark pixel values to
      minimize feature mse distance (of other classes with trigger attached)
      to average feature map of target class.

    See Also:
        * paper: `Latent Backdoor Attacks on Deep Neural Networks`_
        * code: https://github.com/Huiying-Li/Latent-Backdoor
        * website: https://sandlab.cs.uchicago.edu/latent

    Args:
        class_sample_num (int): Sampled input number of each class.
            Defaults to ``100``.
        mse_weight (float): MSE loss weight used in model retraining.
            Defaults to ``0.5``.
        preprocess_layer (str): The chosen layer to calculate feature map.
            Defaults to ``'flatten'``.
        attack_remask_epoch (int): Watermark preprocess optimization epoch.
            Defaults to ``100``.
        attack_remask_lr (float): Watermark preprocess optimization learning rate.
            Defaults to ``0.1``.

    .. _Latent Backdoor Attacks on Deep Neural Networks:
        https://dl.acm.org/doi/10.1145/3319535.3354209
    """
    name: str = 'latent_backdoor'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--class_sample_num', type=int,
                           help='sampled input number of each class '
                           '(default: 100)')
        group.add_argument('--mse_weight', type=float,
                           help='MSE loss weight used in model retraining '
                           '(default: 0.5)')
        group.add_argument('--preprocess_layer',
                           help='the chosen layer to calculate feature map '
                           '(default: "flatten")')
        group.add_argument('--attack_remask_epoch', type=int,
                           help='preprocess optimization epochs')
        group.add_argument('--attack_remask_lr', type=float,
                           help='preprocess learning rate')
        return group

    def __init__(self, class_sample_num: int = 100, mse_weight: float = 0.5,
                 preprocess_layer: str = 'flatten',
                 attack_remask_epoch: int = 100, attack_remask_lr: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)

        self.param_list['latent_backdoor'] = ['class_sample_num', 'mse_weight',
                                              'preprocess_layer', 'attack_remask_epoch', 'attack_remask_lr']
        self.class_sample_num = class_sample_num
        self.mse_weight = mse_weight

        self.preprocess_layer = preprocess_layer
        self.attack_remask_epoch = attack_remask_epoch
        self.attack_remask_lr = attack_remask_lr

        self.avg_target_feats: torch.Tensor = None

    def attack(self, **kwargs):
        print('Sample Data')
        data = self.sample_data()
        print('Calculate Average Target Features')
        self.avg_target_feats = self.get_avg_target_feats(*data['target'])
        print('Preprocess Mark')
        self.preprocess_mark(*data['other'])
        print('Retrain')
        if 'loss_fn' in kwargs.keys():
            kwargs['loss_fn'] = functools.partial(self.loss, loss_fn=kwargs['loss_fn'])
        else:
            kwargs['loss_fn'] = self.loss
        return super().attack(**kwargs)

    def sample_data(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        source_class = self.source_class or list(range(self.dataset.num_classes))
        source_class = source_class.copy()
        if self.target_class in source_class:
            source_class.remove(self.target_class)
        other_x, other_y = [], []
        dataset = self.dataset.get_dataset('train')
        for _class in source_class:
            class_set = self.dataset.get_class_subset(dataset, class_list=[_class])
            _input, _label = sample_batch(class_set, batch_size=self.class_sample_num)
            other_x.append(_input)
            other_y.append(_label)
        other_x = torch.cat(other_x)
        other_y = torch.cat(other_y)
        target_set = self.dataset.get_class_subset(dataset, class_list=[self.target_class])
        target_x, target_y = sample_batch(target_set, batch_size=self.class_sample_num)
        data = {'other': (other_x, other_y),
                'target': (target_x, target_y)}
        return data

    def get_avg_target_feats(self, target_input: torch.Tensor, target_label: torch.Tensor):
        with torch.no_grad():
            if self.dataset.data_shape[1] > 100:
                dataset = TensorDataset(target_input, target_label)
                loader = torch.utils.data.DataLoader(
                    dataset=dataset, batch_size=self.dataset.batch_size // max(env['num_gpus'], 1),
                    num_workers=0, pin_memory=True)
                feat_list = []
                for data in loader:
                    target_x, _ = self.model.get_data(data)
                    feat_list.append(self.model.get_layer(
                        target_x, layer_output=self.preprocess_layer).detach().cpu())
                avg_target_feats = torch.cat(feat_list).mean(dim=0)
                avg_target_feats = avg_target_feats.to(target_x.device)
            else:
                target_input, _ = self.model.get_data((target_input, target_label))
                avg_target_feats = self.model.get_layer(
                    target_input, layer_output=self.preprocess_layer).mean(dim=0)
        return avg_target_feats.detach()

    def preprocess_mark(self, other_input: torch.Tensor, other_label: torch.Tensor):
        other_set = TensorDataset(other_input, other_label)
        other_loader = self.dataset.get_dataloader(mode='train', dataset=other_set, num_workers=0)

        atanh_mark = torch.randn_like(self.mark.mark[:-1], requires_grad=True)
        self.mark.mark[:-1] = tanh_func(atanh_mark)
        optimizer = optim.Adam([atanh_mark], lr=self.attack_remask_lr)
        optimizer.zero_grad()

        for _ in range(self.attack_remask_epoch):
            for (_input, _label) in other_loader:
                poison_input = self.add_mark(to_tensor(_input))
                loss = self.loss_mse(poison_input)
                loss.backward(inputs=[atanh_mark])
                optimizer.step()
                optimizer.zero_grad()
                self.mark.mark[:-1] = tanh_func(atanh_mark)
        atanh_mark.requires_grad_(False)
        self.mark.mark.detach_()

    # -------------------------------- Loss Utils ------------------------------ #
    def loss(self, _input: torch.Tensor, _label: torch.Tensor,
             loss_fn: Callable[..., torch.Tensor] = None,
             **kwargs) -> torch.Tensor:
        loss_fn = loss_fn if loss_fn is not None else self.model.loss
        loss_ce = loss_fn(_input, _label, **kwargs)
        poison_input = self.add_mark(_input)
        loss_mse = self.loss_mse(poison_input)
        return loss_ce + self.mse_weight * loss_mse

    def loss_mse(self, poison_input: torch.Tensor) -> torch.Tensor:
        poison_feats = self.model.get_layer(poison_input, layer_output=self.preprocess_layer)
        return F.mse_loss(poison_feats, self.avg_target_feats)
