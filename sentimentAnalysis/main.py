# coding=utf-8
# @author Cz
#

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import re

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import gensim
from gensim import corpora, models, similarities
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

import pickle
import joblib
import codecs
from datetime import  datetime
import multiprocessing

import sentimentAnalysis.utils.common as um

##################
# 处理日期的工具
##################
def f(str):
    if(str[0] == '0'):
        return int(str[1])
    else:
        return int(str)

def process():
    ##################
    # 预处理数据
    ##################
    data = pd.read_csv('./data/data.csv')
    wod = um.wod('./data/stopwords_hit.txt')
    data['cut_text'] = data['text'].apply(wod.so)
    data.to_csv('./data/data_test_process.csv', encoding='utf-8', index=False)

def draw():

    pro_data = pd.read_csv('./data/data_test_process.csv')

    ##################
    # 发评论数论与时段的统计
    ##################

    all_time = pro_data['time'].apply(lambda time: re.search(r"(\d{2}):\d{2}:\d{2}", time).group(1))
    time_freq = [0 for i in range(24)]
    def g(h):
        time_freq[h] += 1
    all_time.apply(f).apply(g)
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.figsize'] = (12, 5)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.bar(range(len(time_freq)), time_freq, color='#41B6E6')
    ax1.set_xlabel('发评论时间')
    ax1.set_ylabel('评论数量')
    ax1.set_title('各时间段留言数量')
    for i, (_x, _y) in enumerate(zip(range(len(time_freq)), time_freq)):
        plt.text(_x, _y, time_freq[i], color="#001871")
    plt.savefig('./picture/time_count.png',dpi=900, bbox_inches='tight')

    ##################
    # 关键词词频统计 前50个
    ##################

    wod = um.wod('stopwords_hit.txt')
    sw = wod.get_stop_words()
    tfidf_m = CountVectorizer(max_features=50, max_df=0.8,
                              min_df=3,
                              stop_words=sw, ngram_range=(1,2)).fit(np.array(pro_data['cut_text']))
    count = tfidf_m.fit_transform(pro_data['cut_text'])
    try:
        with open('output_model/tf_idf_vector.model', 'wb') as f:
            c = {
                "tfidf": count
            }
            pickle.dump(c, f)
            print('saved done')
    except:
        f.close()
    # print(len(tfidf_m.get_feature_names()))
    # print(tfidf_m.vocabulary_)
    def d(t):
        temp = [i for i in t.split(' ') if tfidf_m.vocabulary_.get(i)]
        return temp
    sum = np.sum(count, axis=0)
    res = {}
    for key, value in  tfidf_m.vocabulary_.items():
        res[key] = sum[value]
    res_sort = sorted(res.items(), key=lambda k: k[1], reverse=True)
    # print(res)

    #################
    # 共现词统计
    # node: [关键词1, 关键词2......] 50个
    # edge: 每个节点与其他节点的关系 {node1: {node2: weight}}
    ##################

    relationship = {}
    edge = {}
    cut = pro_data['cut_text'].apply(d)
    for i in cut:
        if len(i) == 0:
            continue
        for index, w in enumerate(i):
            if w not in edge:
                edge[w] = {}
            if w == i[-1:]:
                break
            nex = i[index+1:]
            for d in nex:
                if d == w: continue
                if d not in edge[w]:
                    edge[w][d] = 1
                else:
                    edge[w][d] +=1
    # print(edge)
    with codecs.open('./data/co-occurence_node.csv', 'w', 'utf-8-sig') as f:
        f.write('Id,Label,Weight \r\n')
        for n, t in res.items():
            f.write(n + ',' + n + ',' +str(t) + '\r\n')
    w_weight = {}
    w_word = {}
    with codecs.open('./data/co-occurence_edge.csv', 'w', 'utf-8-sig') as f:
        f.write('Source,Target,Weight\r\n')
        for n, e in edge.items():
            count = 0
            w_a = []
            for v, w in e.items():
                if w > 3:
                    count += w
                    w_a.append(v)
                    f.write(n + ',' + v + ',' + str(w) + '\r\n')
            w_word[n] = w_a
            w_weight[n] = count
    w_weight = sorted(w_weight.items(), key=lambda k: k[1], reverse=True)
    # print(w_word)
    print(w_weight)
    for i in w_weight[:10]:
        print('{} \t 共现词汇={}'.format(i[0], w_word[i[0]]))


#############
# 情感正负向分析 情感得分
#############

from sentimentAnalysis.utils.common import LSTM
def analysis():

    data = pd.read_csv('data/data_test_process.csv')
    lstm = LSTM(word_dict='./data/dict_word.pk',
                label_dict='./data/dict_label.pk',
                model='./output_model/lstm_2_r.h5')
    data['score'] = data['text'].apply(lstm.predict)
    print('done')
    print(data.head(10))
    data.to_csv('./data/data_test_with_score.csv', index=False, encoding='utf-8-sig')
    # print('saved')
    # print(data['score'].value_counts())
    # print(data['score'].value_counts()[1])

    data = pd.read_csv('./data/data_test_with_score.csv')
    data['hour'] = data['time'].apply(lambda time: re.search(r"(\d{2}):\d{2}:\d{2}", time).group(1)).apply(f)
    time_score = [0 for i in range(24)]
    for i in range(24):
        temp = data[data['hour'] == i]
        l = len(temp)
        p = temp[temp['score'] == 1]['score'].sum() / l *0.4
        n = temp[temp['score'] == -1]['score'].sum() / l *1.6
        time_score[i] = np.around(n+p, decimals=3)

    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.figsize'] = (12, 5)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('发评论时间')
    ax1.set_ylabel('情感均分')
    ax1.set_title('各时间段情感均分')
    ax1.set_ylim(-0.25,0.1)
    plt.plot(range(len(time_score)), time_score, color="#41B6E6")
    plt.savefig('../picture/time_score.png',dpi=900, bbox_inches='tight')


def main():
    process()
    draw()
    analysis()




if __name__ == '__main__':

    main()




