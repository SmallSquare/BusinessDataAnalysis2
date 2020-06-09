import csv


def save_csv(commentlist):
    headers = ['id', 'text', 'time', 'name']

    rows = []

    for comment in commentlist:
        rows.append([comment['id'], comment['text'], comment['time'], comment['name']])

    print(rows)

    with open('data.csv', 'w', encoding='utf-8')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(rows)
