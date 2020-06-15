import csv


def save_csv(commentlist):
    headers = ['id', 'text', 'time', 'name', 'area', 'sex']

    rows = []

    for comment in commentlist:
        rows.append([comment['id'], comment['text'], comment['time'], comment['name'], comment["area"], comment["sex"]])

    print(rows)

    with open('data.csv', 'w', encoding='utf-8')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(rows)
