import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def rangeboundloss(params, lb, ub, factor):
    all_loss = []
    for i in range(len(params)):
        loss=0
        # lb = lb[i]
        # ub = ub[i]
        upper_bound_loss = factor * torch.relu(params[i] - ub)
        lower_bound_loss = factor * torch.relu(lb - params[i])
        loss = loss + upper_bound_loss + lower_bound_loss
        # loss = loss.numpy()
        all_loss.append(loss.item())
    return all_loss

w = torch.rand(3, 200)
wsum = torch.sum(w, dim=0)
loss = rangeboundloss(wsum, 0.95, 1.05, 1)
print(wsum.numpy())
print(loss)

plt.plot(wsum.numpy(), loss, 'o')
plt.ylabel('loss')
plt.xlabel('wsum')
plt.savefig('loss_fun.png')