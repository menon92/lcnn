from collections import OrderedDict, defaultdict

from torchvision.utils import save_image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lcnn.config import M


class MultitaskHead(nn.Module):
    def __init__(self, input_channels, num_class):
        super(MultitaskHead, self).__init__()

        m = int(input_channels / 4)
        heads = []
        for output_channels in sum(M.head_size, []):
            heads.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, m, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(m, output_channels, kernel_size=1),
                )
            )
        self.heads = nn.ModuleList(heads)
        assert num_class == sum(sum(M.head_size, []))

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=1)


class MultitaskLearner(nn.Module):
    def __init__(self, backbone):
        super(MultitaskLearner, self).__init__()
        self.backbone = backbone
        head_size = M.head_size
        self.num_class = sum(sum(head_size, []))
        self.head_off = np.cumsum([sum(h) for h in head_size])

    def forward(self, input_dict):
        print(f"input_dict keys: {input_dict.keys()}")
        image = input_dict["image"]
        save_image(image[0], 'logs/input_image.jpg')
        outputs, feature = self.backbone(image)
        result = {"feature": feature}
        batch, channel, row, col = outputs[0].shape

        T = input_dict["target"].copy()
        n_jtyp = T["jmap"].shape[1]
        print('n_jtyp:', n_jtyp)
        print(f'T["jmap"] shape: {T["jmap"].shape}')
        save_image(T["jmap"][0], 'logs/target_jmap.jpg')

        # switch to CNHW
        for task in ["jmap"]:
            print(f"{task} before permutate: {T[task].shape}")
            T[task] = T[task].permute(1, 0, 2, 3)
            print(f"{task} after permutate: {T[task].shape}")
        for task in ["joff"]:
            print(f"{task} before permutate: {T[task].shape}")
            T[task] = T[task].permute(1, 2, 0, 3, 4)
            print(f"{task} after permutate: {T[task].shape}")

        print("T[lmap]", T['lmap'].shape)
        save_image(T['lmap'][0], 'logs/target_lmap.jpg')
        save_image(T["joff"][0][0][0], 'logs/target_joff.jpg')
        offset = self.head_off # [2 3 5]
        loss_weight = M.loss_weight
        losses = []
        print('offset:', offset)
        for stack, output in enumerate(outputs):
            # 5 x N x H X W
            output = output.transpose(0, 1).reshape([-1, batch, row, col]).contiguous()
            print(f">> stack: {stack}, output: {output.shape}")
            jmap = output[0 : offset[0]].reshape(n_jtyp, 2, batch, row, col)
            lmap = output[offset[0] : offset[1]].squeeze(0)
            joff = output[offset[1] : offset[2]].reshape(n_jtyp, 2, batch, row, col)
            print("pred jmap", jmap.shape, 'pred lmap', lmap.shape, 'pred joff', joff.shape)
            if stack == 0:
                result["preds"] = {
                    "jmap": jmap.permute(2, 0, 1, 3, 4).softmax(2)[:, :, 1],
                    "lmap": lmap.sigmoid(),
                    "joff": joff.permute(2, 0, 1, 3, 4).sigmoid() - 0.5,
                }
                print(f'result["preds"] for stack {stack}')
                for task, pred in result['preds'].items():
                    print(task, pred.shape)
                if input_dict["mode"] == "testing":
                    return result

            L = OrderedDict()
            print('-'*50)
            print('jmap input | target')
            print(jmap[0].shape, "|", T["jmap"][0].shape)
            L["jmap"] = sum(
                cross_entropy_loss(jmap[i], T["jmap"][i]) for i in range(n_jtyp) # n_jtype {R, G, B} or gray
            )
            print("lmap input | target")
            print(lmap.shape, "|", T["lmap"].shape)
            L["lmap"] = (
                F.binary_cross_entropy_with_logits(lmap, T["lmap"], reduction="none")
                .mean(2)
                .mean(1)
            )
            print(f"loss L['lmap'] {L['lmap']}")
            print("joff input | target")
            print(joff[0, 0].shape, "|", T["joff"][0, 0].shape)
            L["joff"] = sum(
                sigmoid_l1_loss(joff[i, j], T["joff"][i, j], -0.5, T["jmap"][i])
                for i in range(n_jtyp)
                for j in range(2)
            )
            for loss_name in L:
                L[loss_name].mul_(loss_weight[loss_name])
            losses.append(L)
        result["losses"] = losses
        return result


def l2loss(input, target):
    return ((target - input) ** 2).mean(2).mean(1)


def cross_entropy_loss(logits, positive):
    print('---')
    print("cross entropy loss ....")
    print(f"logis shape: {logits.shape}")
    print(f"positive shape {positive.shape}")
    nlogp = -F.log_softmax(logits, dim=0)
    print(f'nlogp {nlogp.shape}, nlogp[1] {nlogp[1].shape}')
    print('-' * 30)
    return (positive * nlogp[1] + (1 - positive) * nlogp[0]).mean(2).mean(1)


def sigmoid_l1_loss(logits, target, offset=0.0, mask=None):
    logp = torch.sigmoid(logits) + offset
    loss = torch.abs(logp - target)
    if mask is not None:
        w = mask.mean(2, True).mean(1, True)
        w[w == 0] = 1
        loss = loss * (mask / w)

    return loss.mean(2).mean(1)
