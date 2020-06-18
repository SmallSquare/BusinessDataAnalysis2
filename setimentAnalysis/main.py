# coding=utf-8
# @author Cz
#
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import codecs
import re
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pickle
import joblib

from datetime import  datetime
import multiprocessing

import gensim
from gensim import corpora, models, similarities
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

import setimentAnalysis.utils.common as um


def f(str):
    if(str[0] == '0'):
        return int(str[1])
    else:
        return int(str)

def train_w2v_test():
    wod = um.wod('hit_stopwords.txt')
    sw = wod.get_stop_words()
    pro_data = pd.read_csv('data/process-data.csv')
    # text = pro_data['text'].apply(wod.so).values
    # # with codecs.open('word.txt', 'w', 'utf-8-sig'):
    # # d = []
    # # for i in text:
    # #     d.append(i)
    # print('train start')
    # model = Word2Vec(sentences=text, size=1000, window=5, min_count=2,)
    # model.save('w2v.model')
    # print('save done')
    # wm = Word2Vec.load('w2v.model')
    # print('start to test')
    # test_words = ['活着', '朋友', '分手', '工作', '睡眠', '离开']
    # for i in range(6):
    #     res = wm.most_similar(test_words[i], topn=10)
    #     print(test_words[i])
    #     print(res)
def lda():
    pro_data = pd.read_csv('data/process-data.csv')
    wod = um.wod('hit_stopwords.txt')
    text = np.array(pro_data['text'].apply(wod.so))
    dic = corpora.Dictionary(text)
    corpus = [dic.doc2bow(t) for t in text]
    print('start to tarin')
    lda = gensim.models.ldamodel.LdaModel(
                    corpus=corpus,
                    num_topics=10,
                    id2word=dic,
                    iterations=100
    )
    print('done')
    print(lda.print_topic(0, topn=20))
    print(lda.print_topic(1, topn=20))
    print(lda.print_topic(2, topn=20))
    print(lda.print_topic(3, topn=20))
    joblib.dump(lda, 'output_model/top_model.pkl')

def process():
    data = pd.read_csv('./data/data.csv')
    wod = um.wod('./data/hit_stopwords.txt')
    data['cut_text'] = data['text'].apply(wod.so)
    data.to_csv('./data/process-data.csv', encoding='utf-8', index=False)

def draw():
    pro_data = pd.read_csv('./data/process-data.csv')
    ########## 发评论数论与时段的统计
    # all_time = pro_data['time'].apply(lambda time: re.search(r"(\d{2}):\d{2}:\d{2}", time).group(1))
    # time_freq = [0 for i in range(24)]
    # def g(h):
    #     time_freq[h] += 1
    # all_time.apply(f).apply(g)
    # matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    # matplotlib.rcParams['font.family'] = 'sans-serif'
    # matplotlib.rcParams['axes.unicode_minus'] = False
    # plt.rcParams['figure.figsize'] = (12, 5)
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # lx = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14',
    #       '15','16','17','18','19','20','21','22','23']
    # ax1.bar(range(len(time_freq)), time_freq, color='#41B6E6')
    # ax1.set_xlabel('发评论时间')
    # ax1.set_ylabel('评论数量')
    # ax1.set_title('各时间段留言数量')
    # for i, (_x, _y) in enumerate(zip(range(len(time_freq)), time_freq)):
    #     plt.text(_x, _y, time_freq[i], color="#001871")
    # plt.xticks(range(len(time_freq)), lx)
    # plt.savefig('time_count.png',dpi=900, bbox_inches='tight')


    ######## 关键词词频统计 前50个
    # wod = um.wod('hit_stopwords.txt')
    # sw = wod.get_stop_words()
    # tfidf_m = CountVectorizer(max_features=50, max_df=0.8,
    #                           min_df=3,
    #                           stop_words=sw, ngram_range=(1,2)).fit(np.array(pro_data['cut_text']))
    # count = tfidf_m.fit_transform(pro_data['cut_text'])
    # try:
    #     with open('output_model/tf_idf_vector.model', 'wb') as f:
    #         c = {
    #             "tfidf": count
    #         }
    #         pickle.dump(c, f)
    #         print('saved done')
    # except:
    #     f.close()

    # print(len(tfidf_m.get_feature_names()))
    # print(tfidf_m.vocabulary_)
    # def d(t):
    #     temp = [i for i in t.split(' ') if tfidf_m.vocabulary_.get(i)]
    #     return temp
    # sum = np.sum(count, axis=0)
    # res = {}
    # for key, value in  tfidf_m.vocabulary_.items():
    #     res[key] = sum[value]
    #
    # res_sort = sorted(res.items(), key=lambda k: k[1], reverse=True)
    #
    # # print(res)
    #
    #############    共现词统计
    # relationship = {}
    # # node = {}
    # edge = {}
    # cut = pro_data['cut_text'].apply(d)
    # for i in cut:
    #     if len(i) == 0:
    #         continue
    #     for index, w in enumerate(i):
    #         # if w not in node:
    #         #     node[w] = 1
    #         # else:
    #         #     node[w] += 1
    #         if w not in edge:
    #             edge[w] = {}
    #         if w == i[-1:]:
    #             break
    #         nex = i[index+1:]
    #         for d in nex:
    #             if d == w: continue
    #             if d not in edge[w]:
    #                 edge[w][d] = 1
    #             else:
    #                 edge[w][d] +=1
    #
    # # print(edge)
    # with codecs.open('node.csv', 'w', 'utf-8-sig') as f:
    #     f.write('Id,Label,Weight \r\n')
    #     for n, t in res.items():
    #         f.write(n + ',' + n + ',' +str(t) + '\r\n')
    # w_weight = {}
    # w_word = {}
    # with codecs.open('edge.csv', 'w', 'utf-8-sig') as f:
    #     f.write('Source,Target,Weight\r\n')
    #     for n, e in edge.items():
    #         count = 0
    #         w_a = []
    #         for v, w in e.items():
    #             if w > 3:
    #                 count += w
    #                 w_a.append(v)
    #                 f.write(n + ',' + v + ',' + str(w) + '\r\n')
    #         w_word[n] = w_a
    #         w_weight[n] = count
    # w_weight = sorted(w_weight.items(), key=lambda k: k[1], reverse=True)
    # # print(w_word)
    # print(w_weight)
    # for i in w_weight[:10]:
    #     print('{} \t 共现词汇={}'.format(i[0], w_word[i[0]]))


####todo 情感正负向分析 情感得分
def analysis():
    return



def main():
    # process()
    draw()
    # analysis()


    # train_w2v_test()
    # lda()


if __name__ == '__main__':
    # print(int())
    main()




