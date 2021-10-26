import numpy as np
import os
import sys
from reprod_log import ReprodDiffHelper
from reprod_log import ReprodLogger

DIR = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(DIR, '../../'))

# import torch RotNet
from pytorchRotNet.architectures import NetworkInNetwork as NetworkInNetwork
from pytorchRotNet.architectures import NonLinearClassifier

# import torch RotNet
from paddleRotNet.architectures import NetworkInNetwork as ext
from paddleRotNet.architectures import NonLinearClassifier as cla

import torch
import paddle

def main():
    reprod_log_1 = ReprodLogger()
    reprod_log_2 = ReprodLogger()
    # ----------------------------------------数据----------------------------------------
    np.random.seed(22)
    data_ext_1 = np.random.rand(128, 3, 32, 32).astype(np.float32)

    target_cla_1 = np.random.randint(0, 10, (128, 1))

    criterion_torch = torch.nn.CrossEntropyLoss()
    criterion_paddle = paddle.nn.CrossEntropyLoss()
    x, y = data_ext_1, target_cla_1
    # paddle
    paddle_x = paddle.to_tensor(x)
    paddle_y = paddle.to_tensor(y)

    predicted = model_ext_paddle(paddle_x, ['conv2'])
    predicted = model_cla_paddle(predicted)
    # predicted = paddle.transpose(predicted, perm=[0, 2, 3, 1])
    paddle_loss = criterion_paddle(predicted, paddle_y)
    loss_paddle = paddle_loss.numpy()
    # torch
    torch_x = torch.from_numpy(x)
    torch_y = torch.from_numpy(y).view(-1)

    predicted = model_ext_pytorch(torch_x, ['conv2'])
    predicted = model_cla_pytorch(predicted)
    # predicted = torch.transpose(predicted, perm=[0, 2, 3, 1])
    torch_loss = criterion_torch(predicted, torch_y)
    loss_torch = torch_loss.detach().numpy()
    print(loss_torch)
    print(loss_paddle)
    # torch log
    reprod_log_1.add("model_loss", loss_torch)
    reprod_log_1.save("net_pytorch.npy")
    # paddle log
    reprod_log_2.add("model_loss", loss_paddle)
    reprod_log_2.save("net_paddle.npy")


def check():
    diff_helper = ReprodDiffHelper()
    info1 = diff_helper.load_info("./net_pytorch.npy")
    info2 = diff_helper.load_info("./net_paddle.npy")

    diff_helper.compare_info(info1, info2)
    diff_helper.report(
        diff_method="mean", diff_threshold=1e-6, path="./diff-model.txt")


if __name__ == "__main__":
    get()
    trans()
    main()
    check()
