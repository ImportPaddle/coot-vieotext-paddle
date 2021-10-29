"""
Optimizers.
"""

import math
from typing import Dict, Iterable

import paddle
from paddle.optimizer import Adam, AdamW
from paddle.optimizer import Optimizer

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
        self.weight_decay: paddle.Tensor = config.pop("weight_decay")
        self.weight_decay_for_bias: bool = config.pop("weight_decay_for_bias")
        self.momentum: float = config.pop("momentum")
        self.sgd_nesterov: bool = config.pop("sgd_nesterov")
        self.adam_beta2: float = config.pop("adam_beta2")
        self.adam_eps: float = config.pop("adam_eps")
        self.adam_amsgrad: bool = config.pop("adam_amsgrad")
        self.radam_degentosgd: bool = config.pop("radam_degentosgd")
        self.lr_decay_mult: bool = config.pop("lr_decay_mult")


def make_optimizer(cfg: OptimizerConfig, params: Iterable[paddle.Tensor]) -> Optimizer:
    """
    Initializer optimizer given some configuration and parameters.

    Args:
        cfg: Optimizer type and hyperparameter configuration.
        params: Parameters to optimizer.

    Returns:
        Normalization function class.
    """
    # print('params:', params, 'params len:', len(params))

    if cfg.name == OptimizerConst.ADAM:
        optimizer: Optimizer = Adam(parameters=params, learning_rate=cfg.lr, beta1=cfg.momentum, beta2=cfg.adam_beta2, epsilon=cfg.adam_eps,
                                    weight_decay=cfg.weight_decay)
            # , amsgrad=cfg.adam_amsgrad
    elif cfg.name == OptimizerConst.RADAM:
        optimizer = Adam(parameters=params, learning_rate=cfg.lr, beta1=cfg.momentum, beta2=cfg.adam_beta2, epsilon=cfg.adam_eps,
                          weight_decay=paddle.to_tensor(cfg.weight_decay))
        """
         def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 parameters=None,
                 weight_decay=0.01,
                 apply_decay_param_fun=None,
                 grad_clip=None,
                 lazy_mode=False,
                 multi_precision=False,
                 name=None):
        """
        # optimizer = RAdam(params=params, lr=cfg.lr, betas=(cfg.momentum, cfg.adam_beta2), eps=cfg.adam_eps,
        #                   weight_decay=cfg.weight_decay,
        #                   degenerated_to_sgd=cfg.radam_degentosgd)
    else:
        raise NotImplementedError(f"Unknown optimizer {cfg.name}")

    # apply special lr / weight decay if given by the model.
    lr = cfg.lr
    wd = cfg.weight_decay
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr * param_group['lr_mult']
    #     param_group['weight_decay'] = wd * param_group['decay_mult']

    return optimizer


# ---------- Module implementation. ----------

