# coding=utf-8

# code by CZ.
import  csv
from csv import  DictWriter
import pandas  as pd
import demjson
import numpy as np
import jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from gensim.models import word2vec
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.nlp.v20190408 import nlp_client, models

from aip import AipNlp
class wod(object):
    def __init__(self, word):
        self.word = word
        self.stop_words= []
        self.get_stop_words()

    def get_stop_words(self):
        with open(self.word, encoding='utf-8') as f:
            stopwords = f.read()
        stopwords_list = stopwords.split('\n')
        stopwords = [i for i in stopwords_list]
        self.stop_words = stopwords

    def so(self, sentence):
        seg_list = jieba.cut(sentence, cut_all=False)
        temp = " ".join(seg_list).split()
        out_sentences = []
        for w in temp:
            if w not in self.stop_words:
                out_sentences.append(w)
        return list(out_sentences)


def chinese_word_cut(mytext):
    return " ".join(jieba.cut(mytext))


