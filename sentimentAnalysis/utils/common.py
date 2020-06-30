# coding=utf-8
# code by CZ.

import pandas as pd
import demjson
import numpy as np
import jieba
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


#############
# 分词，去停用词
#############
class wod(object):
    def __init__(self, word):
        self.word = word
        self.stop_words = []
        self.get_stop_words()

    def get_stop_words(self):
        with open(self.word, encoding='utf-8') as f:
            stopwords = f.read()
        stopwords_list = stopwords.split('\n')
        stopwords = [i for i in stopwords_list]
        self.stop_words = stopwords

    def so(self, sentence):
        seg_list = jieba.cut(sentence, cut_all=False)
        # print("seg//////", seg_list)
        temp = " ".join(seg_list).split()
        # print(temp)
        out_sentences = []
        for w in temp:
            if w not in self.stop_words:
                out_sentences.append(w)
        return out_sentences


#############
# 使用LSTM模型
#############
class LSTM(object):
    def __init__(self, word_dict, label_dict, model):
        """
        :param word_dict:  path
        :param label_dict:  path
        :param model:  lstm model
        """
        self.word_dict = word_dict
        self.label_dict = label_dict
        self.model_path = model
        self.load()
        print('load word and label dict and lstm model')

    def load(self):
        with open(self.word_dict, 'rb') as f:
            self.word_dictionary = pickle.load(f)
        with open(self.label_dict, 'rb') as f:
            output_dictionary = pickle.load(f)
        self.label_dict = {v: k for k, v in output_dictionary.items()}
        self.model = load_model(self.model_path)

    def predict(self, text, input_shape=180):
        # print(text)
        try:
            x = [[self.word_dictionary[word] for word in text]]
            x = pad_sequences(maxlen=input_shape, sequences=x, padding='post', value=0)
            y_predict = self.model.predict(x)
            res = self.label_dict[np.argmax(y_predict)]
            return int(res)
        except KeyError as err:
            # print("您输入的句子有汉字不在词汇表中，请重新输入")
            # print("不在词汇表中的单词为：%s." % err)
            return 1
