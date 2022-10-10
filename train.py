import torch.cuda
import torch.nn as nn
from torch.optim import Optimizer


def train_one_epoch(model: nn.Module, data_loader, optimizer: Optimizer, loss_func):
    model.train()
    optimizer.zero_grad()
    loss_total = 0.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for word_list, word_len, label in data_loader:
        word_list = word_list.to(device)
        word_len = word_len.to(device)
        label = label.to(device)
        output = model(word_list, word_len)
        loss = loss_func(output, label)
        loss_total += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss_total / len(data_loader)


def eval_one_epoch(model: nn.Module, data_loader, loss_func):
    loss_total = 0.0
    right_num = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    for word_list, word_len, label in data_loader:
        word_list = word_list.to(device)
        word_len = word_len.to(device)
        label = label.to(device)
        output = model(word_list, word_len)
        loss = loss_func(output, label)
        for i in range(output.size(0)):
            if torch.argmax(output[i]) == label[i]:
                right_num += 1
        loss_total += loss
    return loss_total / len(data_loader), right_num
