# -*- coding : utf-8-*-
import jieba
import json
from gensim.models import Word2Vec, KeyedVectors
import multiprocessing
import torch
import torch.nn as nn


def load_embedding(path):
    wvmodel = KeyedVectors.load_word2vec_format(path, binary=True, unicode_errors='ignore')
    idx_to_word = wvmodel.index_to_key
    word_to_idx = {wvmodel.index_to_key[idx]: idx + 1 for idx in range(len(idx_to_word))}
    word_to_idx["unk"] = 0
    embedding_size = 100
    weight = torch.zeros(len(idx_to_word) + 1, embedding_size)
    for i in range(len(idx_to_word)):
        weight[i + 1, :] = torch.from_numpy(wvmodel.get_vector(wvmodel.index_to_key[i]))
    # embed
    embedding = nn.Embedding.from_pretrained(weight)
    return embedding, word_to_idx


def main():
    path = "./data/train_data/train_data.json"
    data_list = json.load(open(path, encoding='utf-8'))
    sentences = []
    max_len = 0
    for data_dict in data_list:
        sentence = jieba.lcut(data_dict["content"])
        if max_len < len(sentence):
            max_len = len(sentence)
        sentences.append(sentence)
    print(max_len)
    model = Word2Vec(sentences, workers=multiprocessing.cpu_count())
    model.wv.save_word2vec_format("embedding.bin", binary=True)


if __name__ == '__main__':
    main()
    # load_embedding("embedding.bin")
