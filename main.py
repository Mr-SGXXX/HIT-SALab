import random
import logging
import os
import torch
import numpy as np
from train import train_one_epoch, eval_one_epoch
from dataset import MsgDataset
from gen_embedding import load_embedding
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from model import BiLSTM
import torch.optim.lr_scheduler as lr_scheduler
import csv


def train(epoch_num, cur_epoch, model: torch.nn.Module, msg_dir, word_to_idx, learn_rate=1e-3, logger=None):
    # 数据集加载
    train_dataset = MsgDataset(msg_dir, word_to_idx, 600, 0)
    train_dataloader = DataLoader(train_dataset, 64, True)
    eval_dataset = MsgDataset(msg_dir, word_to_idx, 600, 1)
    eval_dataloader = DataLoader(eval_dataset, 64, True)

    optimizer = optim.Adam(model.parameters(), lr=learn_rate, weight_decay=1e-5)
    loss_func = nn.CrossEntropyLoss()
    step_lr = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    for epoch in range(cur_epoch, epoch_num):
        loss = train_one_epoch(model, train_dataloader, optimizer, loss_func)
        if logger is None:
            print(f"Epoch: {epoch + 1}/{epoch_num}\tLoss:{loss}")
        else:
            logger.info(f"Epoch: {epoch + 1}/{epoch_num}\tLoss:{loss}")
        if epoch % 10 == 9:
            # 保存模型参数
            torch.save(model.state_dict(), "./model_weight.pth")
            loss, right_num = eval_one_epoch(model, eval_dataloader, loss_func)
            step_lr.step()
            if logger is None:
                print(f"Eval Loss = {loss}\t Eval Accuracy = {right_num / len(eval_dataset)}")
            else:
                logger.info(f"Eval Loss = {loss}\t Eval Accuracy = {right_num / len(eval_dataset)}")


def model_test(model: torch.nn.Module, msg_dir, word_to_idx):
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_dataset = MsgDataset(msg_dir, word_to_idx, 750, 2)
    test_dataloader = DataLoader(test_dataset, 64, False)
    rst = []
    for word_list, word_len in test_dataloader:
        word_list = word_list.to(device)
        word_len = word_len.to(device)
        output = model(word_list, word_len)
        for i in range(output.size(0)):
            rst.append(int(torch.argmax(output[i])))
    return rst


def main():
    # 设置日志
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler("./train_log.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s]-[%(levelname)s]: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # 固定随机种子
    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    np.random.seed(10)
    random.seed(10)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding, word_to_idx = load_embedding("./embedding.bin")
    model = BiLSTM(embedding, 256, 128, 2).to(device)

    if os.path.exists("./model_weight.pth"):
        try:
            model.load_state_dict(torch.load("./model_weight.pth"))
        except:
            logger.info("Model Weight Load Failed")
    train(500, 0, model, "./data", word_to_idx, logger=logger)
    result = model_test(model, "./data", word_to_idx)
    writer = csv.writer(open("./result.csv", 'w', encoding='utf-8', newline=''))
    for i, t in enumerate(result):
        writer.writerow([i + 1, t])


if __name__ == "__main__":
    main()
