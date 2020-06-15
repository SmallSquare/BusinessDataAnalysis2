import re
import os
import csv
import jieba
from matplotlib import pyplot as plt
import database


def get_sentence_statistics():
    counts = {}

    for comment in database.get_comments():
        sentence = re.sub('<[^<]+?>', '', comment['text']).replace('\n', '').strip()  # re用来去除html标签

        counts[len(sentence)] = counts.get(len(sentence), 0) + 1

    items = list(counts.items())
    items.sort(key=lambda x: x[0], reverse=True)
    for i in range(30):
        word, count = items[i]
        print("{0:<10}{1:<5}".format(word, count))

    return items


def draw_image(items):
    xs = []
    ys = []
    for item in items:
        print(item)
        xs.append(item[0])
        ys.append(item[1])
    plt.figure(figsize=(30, 20), dpi=120)
    plt.plot(xs, ys)
    # plt.xticks(xs)
    # plt.yticks(range(min(ys), max(ys) + 1))
    # plt.savefig("./result.png")
    plt.show()


if __name__ == '__main__':
    items = get_sentence_statistics()
    draw_image(items)

    headers = ['length', 'count']

    rows = []

    for i in range(len(items)):
        length, count = items[i]
        rows.append([length, count])

    print(rows)

    with open('ss.csv', 'w', encoding='utf-8')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(rows)
