"""
Optimizers.
"""

import math
from typing import Dict, Iterable

import paddle
from paddle.optimizer import Adam, AdamW
from paddle.optimizer import Optimizer
from paddle.optimizer.lr import ReduceOnPlateau
from nntrainer import typext


class OptimizerConst(typext.ConstantHolder):
    """
    Optimizer name constants.
    """
    ADAM = "adam"
    RADAM = "radam"


class OptimizerConfig(typext.ConfigClass):
    """
    Optimizer Configuration Class

    Args:
        config: Configuration dictionary to be loaded, optimizer part.
    """

    def __init__(self, config: Dict) -> None:
        self.name: str = config.pop("name")
        self.lr: float = config.pop("lr")
        self.weight_decay: float = float(config.pop("weight_decay"))
        self.weight_decay_for_bias: bool = config.pop("weight_decay_for_bias")
        self.momentum: float = config.pop("momentum")
        self.sgd_nesterov: bool = config.pop("sgd_nesterov")
        self.adam_beta2: float = config.pop("adam_beta2")
        self.adam_eps: float = config.pop("adam_eps")
        self.adam_amsgrad: bool = config.pop("adam_amsgrad")
        self.radam_degentosgd: bool = config.pop("radam_degentosgd")
        self.lr_decay_mult: bool = config.pop("lr_decay_mult")


def make_optimizer(cfg_optim: OptimizerConfig,
                   cfg_lr,
                   params: Iterable[paddle.Tensor]) -> Optimizer:
    """
    Initializer optimizer given some configuration and parameters.

    Args:
        cfg: Optimizer type and hyperparameter configuration.
        params: Parameters to optimizer.

    Returns:
        Normalization function class.
    """
    # print('params:', params, 'params len:', len(params))
    lr_ROP = ReduceOnPlateau(cfg_optim.lr, mode='min', factor=0.1, patience=cfg_lr.rop_patience, threshold=1e-4,
                             threshold_mode='rel', cooldown=cfg_lr.rop_cooldown, min_lr=0,
                             epsilon=1e-8, verbose=False)

    # if True:
    if cfg_optim.name == OptimizerConst.ADAM:
        optimizer = Adam(parameters=params, learning_rate=lr_ROP, beta1=cfg_optim.momentum,
                         beta2=cfg_optim.adam_beta2, epsilon=cfg_optim.adam_eps,
                         weight_decay=cfg_optim.weight_decay)

    elif cfg_optim.name == OptimizerConst.RADAM:
        optimizer = AdamW(parameters=params, learning_rate=lr_ROP, beta1=cfg_optim.momentum,
                          beta2=cfg_optim.adam_beta2, epsilon=cfg_optim.adam_eps,
                          weight_decay=cfg_optim.weight_decay)
    else:
        raise NotImplementedError(f"Unknown optimizer {cfg_optim.name}")

    # apply special lr / weight decay if given by the model.
    wd = cfg_optim.weight_decay
    for param_group in optimizer._param_groups:
        param_group['weight_decay'] = wd * param_group['decay_mult']

    return optimizer, lr_ROP


