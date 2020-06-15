# coding=utf-8

# Code by SmallSquare, 2020/6.
# Provide an easy access to Mysql.
import sys

import chardet
import pymysql
import private_settings

"""
"private_settings.py" is a python file I created in the same path as this file, and it recorded database password,
 so you should create a same file like that, and make a method to return your password.
 I just don't wanna pull this file to Github.
 private_settings.py should like following:
 
    # coding=utf-8
    # Code by SmallSquare, 2020/6.
    # To save some private settings.
    
    def getMysqlPassword():
        return "123456"
        
"""


def insert_comment(commentlist):
    db = pymysql.connect("localhost", "root", private_settings.getMysqlPassword(), "businessdataanalysis",
                         charset='utf8mb4')
    # Must be 'utf8mb4' to be compatible to the 4个编码的 character.

    # use method cursor() to get a 游标.
    cursor = db.cursor()

    sql = "REPLACE INTO weibocomments(id, text , time, name, area, sex) VALUES (%s, %s, %s, %s, %s, %s)"

    try:
        for comment in commentlist:
            cursor.execute(sql, (
            comment["id"], comment["text"], comment["time"], comment["name"], comment["area"], comment["sex"]))
            print(comment)
        db.commit()
    except Exception as e:
        # rollback when get error
        db.rollback()
        print("Insert ERROR, so rollback.")
        print(e)
        print(sys.exc_info())

    db.close()


def get_comments():
    comment_list = []

    db = pymysql.connect("localhost", "root", private_settings.getMysqlPassword(), "businessdataanalysis",
                         charset='utf8mb4')
    cursor = db.cursor()

    sql = "SELECT * FROM weibocomments"
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
        for row in results:
            id = row[0]
            text = row[1]
            time = row[2]
            name = row[3]
            area = row[4]
            sex = row[5]
            # print("id=%s,text=%s,movie_id=%s" % (id, text, movie_id))
            comment_list.append({"id": id, "text": text, "time": time, "name": name, "area": area, "sex": sex})
    except Exception as e:
        print("Unable to fetch data.")

    db.close()

    return comment_list


def del_all(table):
    db = pymysql.connect("localhost", "root", private_settings.getMysqlPassword(), "businessdataanalysis",
                         charset='utf8')

    # use method cursor() to get a 游标.
    cursor = db.cursor()

    sql = "DELETE FROM " + table

    try:
        cursor.execute(sql)
        db.commit()
    except Exception as e:
        # rollback when get error
        db.rollback()
        print("Delete ERROR, so rollback.")
        print(e)

    db.close()
