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


def getComments(id="3424883176420210", pages=100000, getarea=True):
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
            url = "https://m.weibo.cn/comments/hotflow?id=" + str(id) + "&mid=" + str(id) + "&max_id_type=0"
        else:
            url = "https://m.weibo.cn/comments/hotflow?id=" + str(id) + "&mid=" + str(id) + "&max_id=" + str(
                last_max_id) + "&max_id_type=" + str(last_max_id_type)
            # print(url)

        r = session.get(url, headers=headers, cookies=cookie_dict)
        # print(r.text)

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

            area = "未获取"
            if getarea:
                area = getCommentUserArea(uid)

            if commentbody['user']['gender'] == 'f':
                sex = "女"
            else:
                sex = "男"

            commentList.append(
                {'id': commentbody['id'],
                 'text': commentbody['text'],
                 'time': commentbody['created_at'],
                 'name': commentbody['user']["screen_name"],
                 "area": area,
                 "sex": sex})
            try:
                print(commentbody['id'] + "/" + area + "/" + sex + "/" + commentbody['text'])
            except Exception as e:
                print(e)

        last_max_id = jsondatum['data']['max_id']
        last_max_id_type = jsondatum['data']['max_id_type']

        print("已完成" + str(page) + "页")
        page += 1
        if not getarea:
            time.sleep(random.randint(2, 5))

    return commentList


def getCommentUserArea(uid):
    time.sleep(random.randint(2, 3))

    session = requests.Session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_0) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/71.0.3578.98 Safari/537.36'
    }
    cookies = cookielib.LWPCookieJar("Cookie.txt")
    cookies.load(ignore_discard=True, ignore_expires=True)
    cookie_dict = requests.utils.dict_from_cookiejar(cookies)
    # try:
    #     r2 = session.get("http://weibo.cn/" + str(uid) + "/info", headers=headers, cookies=cookie_dict)
    #     matchObj = re.search('地区:([\u4e00-\u9fa5]*)', r2.text)
    #     area = matchObj.group(1)
    # except Exception as e:
    #     print("re Error.")
    #     print(r2.text)
    #     print(e)

    cid = ""
    r = session.get("http://m.weibo.cn/api/container/getIndex?type=uid&value=" + str(uid), headers=headers)
    jsonObj = r.json()
    try:
        for tab in jsonObj['data']['tabsInfo']['tabs']:
            if tab['id'] == 1:
                cid = tab['containerid']
    except Exception as e:
        print(e)
        print(r.text)
    try:
        r = session.get("https://m.weibo.cn/api/container/getIndex?containerid=" + str(cid) + "_-_INFO",
                        headers=headers)
        jsonObj2 = r.json()
        for card in jsonObj2['data']['cards']:
            for cardgroup in card['card_group']:
                try:
                    if cardgroup['item_name'] == "所在地":
                        return cardgroup['item_content']
                except Exception as e:
                    pass
    except Exception as e:
        print(e)
        print(r.text)


def getUidByName(name):
    session = requests.Session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_0) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/71.0.3578.98 Safari/537.36'
    }
    r = session.get(
        "https://m.weibo.cn/api/container/getIndex?containerid=100103type%3D3%26q%3D" + name + "%26t%3D0&page_type=searchall",
        headers=headers)
    jsonObj = r.json()
    for card in jsonObj['data']['cards']:
        if card['card_type'] == 11:
            return card['card_group'][0]['user']['id']


def getUserWeiboContent(uid):
    try:
        content_list = []
        session = requests.Session()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_0) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/71.0.3578.98 Safari/537.36'
        }
        # First step is to get the cid (container id)
        cid = None
        r = session.get("http://m.weibo.cn/api/container/getIndex?type=uid&value=" + str(uid), headers=headers)
        jsonObj = r.json()
        for tab in jsonObj['data']['tabsInfo']['tabs']:
            if tab['id'] == 2:
                cid = tab['containerid']
        # Then, get the weibo blogs
        r2 = session.get("https://m.weibo.cn/api/container/getIndex?containerid=" + str(cid), headers=headers)
        jsonObj = r2.json()
        for card in jsonObj['data']['cards']:
            content_list.append(re.sub('<[^<]+?>', '', card['mblog']['text'].replace('\n', '').strip()))
        return content_list
    except Exception as e:
        print(e.args)


if __name__ == '__main__':
    pass
    # login("18214888360", "6366565")
    # commentlist = getComments("4515487243886433", 1000, False)
    # print("爬到" + str(len(commentlist)) + "条")
    # util_csv.save_csv(commentlist, "positive_data")
    # database.insert_comment(commentlist)

    # getCommentUserArea(5579896374)
    for content in getUserWeiboContent(6107465978):
        print(content)
