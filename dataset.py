from torch.utils.data import Dataset
import torch
import json
import jieba


class MsgDataset(Dataset):
    """
    消息数据集
    迭代器返回消息词序列及对应评价
    正面评价标签为1    中性评价标签为0    负面评价标签为2
    """

    def __init__(self, dir_path: str, word_to_index, max_len, dataset_type=0):
        """
        标签 0表示中立    1表示正面   2表示负面
        :param dir_path:数据集路径，其目录结构应为；
            |- dir_path
                |- eval_data
                    |- eval_data.json   验证集文件
                |- train_data
                    |- train_data.json  训练集文件
                |- test_data
                    |- test_data.json   测试集文件
        :param word_to_index: 词-序号映射表
        :param max_len: 最大句子长度
        :param dataset_type: 0表示训练集，1表示验证集，其他表示测试集
        """
        super(MsgDataset, self).__init__()
        if dataset_type == 0:
            self.test = False
            file = open(dir_path + "/train_data/train_data.json", mode='r', encoding='utf-8')
        elif dataset_type == 1:
            self.test = False
            file = open(dir_path + "/eval_data/eval_data.json", mode='r', encoding='utf-8')
        else:
            self.test = True
            file = open(dir_path + "/test_data/test_data.json", mode='r', encoding='utf-8')
        self.data_list = json.load(file)
        self.word_to_idx = word_to_index
        self.len = len(self.data_list)
        self.max_len = max_len

    def __getitem__(self, item):
        data_dict = self.data_list[item]
        if not self.test:
            if data_dict["label"] == "positive":
                label = 1
            elif data_dict["label"] == "neutral":
                label = 0
            elif data_dict["label"] == "negative":
                label = 2
            else:
                raise ValueError
        words = [self.word_to_idx[word] if word in self.word_to_idx else self.word_to_idx["unk"] for word in
                 jieba.lcut(data_dict["content"])]
        while True:
            if len(words) < self.max_len:
                words.append(0)
            else:
                break
        if not self.test:
            return torch.LongTensor(words).resize_(self.max_len), torch.tensor(len(words)), label
        else:
            return torch.LongTensor(words).resize_(self.max_len), torch.tensor(len(words))

    def __len__(self):
        return self.len
