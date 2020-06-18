# coding=utf-8
# code by Cz.

import pandas as pd
import numpy as np
import keras
import pickle

from keras.models import Sequential
from keras.models import load_model
from keras.utils import np_utils, plot_model
from keras.preprocessing import text, sequence
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split

from sentimentAnalysis.utils import  common as uc

import matplotlib.pyplot as plt
from matplotlib import font_manager
from itertools import accumulate

class LSTM_U(object):

    def __init__(self, data):
        self.data = pd.read_csv(data, encoding='utf-8-sig' )
        self.model = None
        self.vocab_size = None
        self.X = self.y = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None

    def draw(self):

        ##################
        # 绘图分析数据
        ##################

        font = font_manager.FontProperties(fname='yahei.ttf')
        print(len(self.data['text'][0]))
        def d(text):
            return len(text)
        self.data['length'] = self.data['text'].apply(d)
        len_df = self.data.groupby('length').count()
        # 评论长度
        sen_len = len_df.index.tolist()
        # 评论长度出现频率
        sen_freq = len_df['text'].tolist()
        # 绘制评论长度累积分布函数(CDF)
        sent_pentage_list = [(count / sum(sen_freq)) for count in accumulate(sen_freq)]
        plt.plot(sen_len, sent_pentage_list)
        # 寻找分位点为quantile的句子长度
        quantile = 0.98
        for length, per in zip(sen_len, sent_pentage_list):
            if round(per, 2) == quantile:
                index = length
                break
        print("\n分位点为%s的句子长度:%d." % (quantile, index))
        # 绘制句子长度累积分布函数图
        plt.plot(sen_len, sent_pentage_list)
        plt.hlines(quantile, 0, index, colors="c", linestyles="dashed")
        plt.vlines(index, 0, quantile, colors="c", linestyles="dashed")
        plt.text(0, quantile, str(quantile))
        plt.text(index, 0, str(index))
        plt.xlabel("句子长度", fontproperties=font)
        plt.ylabel("句子长度累积频率", fontproperties=font)
        plt.show()
        plt.close()

    def pro_precess(self):

        ##################
        # 预处理数据
        ##################

        X = self.data['text']
        y = self.data['label']
        #标签：1 轻度/不严重 -1 中度和重度
        # 标签及词汇表
        labels, vocabulary = list(y.unique()), list(X.unique())
        # 构造字符级别的特征
        string = ''
        for word in vocabulary:
            string += word

        vocabulary = set(string)
        # 字典列表
        word_dictionary = {word: i + 1 for i, word in enumerate(vocabulary)}
        with open('../data/dict_word.pk', 'wb') as f:
            pickle.dump(word_dictionary, f)
        inverse_word_dictionary = {i + 1: word for i, word in enumerate(vocabulary)}
        label_dictionary = {label: i for i, label in enumerate(labels)}
        with open('../data/dict_label.pk', 'wb') as f:
            pickle.dump(label_dictionary, f)
        output_dictionary = {i: labels for i, labels in enumerate(labels)}
        self.vocab_size = len(word_dictionary.keys())  # 词汇表大小
        print('vocab size', self.vocab_size)
        self.label_size = len(label_dictionary.keys())  # 标签类别数量
        print('label size', self.label_size)
        # 序列填充，按input_shape填充，长度不足的按0补充
        X = [[word_dictionary[word] for word in sent] for sent in X]
        self.X = sequence.pad_sequences(maxlen=180, sequences=X, padding='post', value=0)
        y = [[label_dictionary[sent]] for sent in y]
        y = [np_utils.to_categorical(label, num_classes=self.label_size) for label in y]
        self.y = np.array([list(_[0]) for _ in y])
        print('X shape', self.X.shape)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                                                            self.X,
                                                            self.y,
                                                            test_size=0.2,
                                                            random_state=22)

    def lstm(self):

        ##################
        # 构建LSTM模型
        ##################

        embed_dim = 20
        lstm_out = 200
        self.model = Sequential()
        self.model.add(Embedding(input_dim=self.vocab_size+1,
                                 output_dim=embed_dim,
                                 input_length=180,
                                 mask_zero=True))
        self.model.add(LSTM(lstm_out, input_shape=(self.X.shape[0], self.X.shape[1])))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=self.label_size, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
        plot_model(self.model, to_file='../picture/model_lstm_3_r.png', show_shapes=True)
        print(self.model.summary())

    def train(self):

        ##################
        # 训练LSTM模型
        ##################

        batch_size = 32
        epochs = 15
        verbose = True
        print('##########################lstm start to train##########################')
        history = self.model.fit(self.X_train, self.y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 verbose=verbose)
        print(history.history.keys())
        fig = plt.figure()
        plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='lower left')
        fig.savefig('../picture/loss_2_r.png')
        fig = plt.figure()
        # plt.plot(history.history['loss'])
        plt.plot(history.history['accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        fig.savefig('../picture/accuracy_2_r.png')

        #self.model.fit(self., Y, batch_size=batch_size, epochs=5, verbose=1)
        #self.model.train_on_batch(x_batch, y_batch)
        #预测
        # classes = model.predict(x_test, batch_size=128)

    def test(self):

        ##################
        # 测试LSTM模型
        ##################

        batch_size = 32
        print('##########################lstm start to test##########################')
        loss_and_metrics = self.model.evaluate(self.X_test, self.y_test, batch_size=batch_size, verbose=True)
        print('loss', loss_and_metrics)

    def save_model(self):

        ##################
        # 保存LSTM模型
        ##################

        print('##########################lstm start to save##########################')
        self.model.save('../output_model/lstm_2_r.h5')
        print('##########################lstm save done##############################')


def main():

    data = '../data/data_train_with_label.csv'
    lstm = LSTM_U(data)
    # lstm.draw()
    lstm.pro_precess()
    lstm.lstm()
    lstm.train()
    lstm.test()
    lstm.save_model()



if __name__ == '__main__':
    main()
