import re
import os
import csv
import jieba
import database


def get_stop_words():
    txt_path = os.path.join(os.path.dirname(__file__), 'stopwords.txt')
    f = open(txt_path)
    data_lists = f.readlines()
    stopwords = set()
    for data in data_lists:
        stopwords.add(data.rstrip('\n'))
    return stopwords


def get_word_statistics():
    total_comment_text = ""
    for comment in database.get_comments():
        total_comment_text += re.sub('<[^<]+?>', '', comment['text']).replace('\n', '').strip()  # re用来去除html标签
    words = jieba.lcut(total_comment_text)
    stop_words = get_stop_words()

    counts = {}
    for word in words:
        if len(word) == 1:
            continue
        elif word in stop_words:
            continue
        else:
            counts[word] = counts.get(word, 0) + 1
    items = list(counts.items())
    items.sort(key=lambda x: x[1], reverse=True)
    for i in range(30):
        word, count = items[i]
        print("{0:<10}{1:<5}".format(word, count))

    return items


if __name__ == '__main__':
    ws = get_word_statistics()

    headers = ['word', 'count']

    rows = []

    for i in range(100):
        word, count = ws[i]
        rows.append([word, count])

    print(rows)

    with open('ws.csv', 'w', encoding='utf-8')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(rows)
