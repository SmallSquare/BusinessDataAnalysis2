# coding=utf-8

# Code by SmallSquare, 2020/6.
# Only for the course design of the Business Data Analysis.
import random
import sys

import requests
import time
import json
import bs4
import re
import database
import util_csv
import spider.get_proxy as get_proxy
import spider.weibo_login
import http.cookiejar as cookielib


def login(usrname, pwd):
    username = usrname
    password = pwd
    cookie_path = "Cookie.txt"
    weibo = spider.weibo_login.WeiboLogin(username, password, cookie_path)
    weibo.login()
    print("Login successfully.")


def getComments(pages=100000):
    """
    This method can get comments.
    :return: commentlist
    """

    commentList = []
    page = 1

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_0) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/71.0.3578.98 Safari/537.36'
    }
    session = requests.Session()

    last_max_id = ""
    last_max_id_type = ""
    cookies = cookielib.LWPCookieJar("Cookie.txt")
    cookies.load(ignore_discard=True, ignore_expires=True)
    cookie_dict = requests.utils.dict_from_cookiejar(cookies)
    while True:
        if pages <= page:
            break

        if last_max_id == "":
            url = "https://m.weibo.cn/comments/hotflow?id=3424883176420210&mid=3424883176420210&max_id_type=0"
        else:
            url = "https://m.weibo.cn/comments/hotflow?id=3424883176420210&mid=3424883176420210&max_id=" + str(
                last_max_id) + "&max_id_type=" + str(last_max_id_type)
            print(url)

        r = session.get(url, headers=headers, cookies=cookie_dict)

        print(r.text)

        try:
            jsondatum = r.json()
        except Exception as e:
            print(sys.exc_info())
            print(r.text)
            print("爬虫遇到错误")
            break

        if jsondatum['ok'] == 0:
            break

        for commentbody in jsondatum['data']['data']:
            uid = commentbody['user']['id']
            area = "其他"
            sex = "无"
            try:
                r2 = session.get("http://weibo.cn/" + str(uid) + "/info", headers=headers, cookies=cookie_dict)
                matchObj = re.search('地区:([\u4e00-\u9fa5]*)', r2.text)
                matchObj2 = re.search('性别:([\u4e00-\u9fa5]*)', r2.text)
                area = matchObj.group(1)
                sex = matchObj2.group(1)
            except Exception as e:
                print("re Error.")
                print(e)

            commentList.append(
                {'id': commentbody['id'],
                 'text': commentbody['text'],
                 'time': commentbody['created_at'],
                 'name': commentbody['user']["screen_name"],
                 "area": area,
                 "sex": sex})

        last_max_id = jsondatum['data']['max_id']
        last_max_id_type = jsondatum['data']['max_id_type']

        page += 1
        print(page)
        time.sleep(random.randint(1, 4))

    return commentList


if __name__ == '__main__':
    login("18214888360", "6366565")
    commentlist = getComments(2000)
    print("爬到" + str(len(commentlist)) + "条")
    util_csv.save_csv(commentlist)
    database.insert_comment(commentlist)
