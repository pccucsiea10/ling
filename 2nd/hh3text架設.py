#!/usr/bin/env python
# coding: utf-8

# In[1]:


from elasticsearch import Elasticsearch
es = Elasticsearch(hosts='140.137.41.81', port=9200)
#es.indices.delete(index='hh3text')


# In[2]:


from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
import operator
import math


# In[3]:


from elasticsearch import Elasticsearch
import json

def create_index(es):
    body = dict()
    body['settings'] = get_setting()
    body['mappings'] = get_mappings()
    print(json.dumps(body)) #可以用json.dumps輸出來看格式有沒又包錯
    es.indices.create(index='hh3text', body=body)

def get_setting():
    settings = {
        "index": {
            "number_of_shards": 3,
            "number_of_replicas": 2
        }
    }
    return settings

def get_mappings():
    mappings = {
        "properties": {
            #編號
            "time Number": {
                "type": "integer"
            },
            #編號
            "name": {
                "type": "keyword"
            },
            #影片類別
            "subject": {
                "type": "keyword"
            },
            #平均
            "average": {
                "type": "float"
            },
            #一階差
            "first order difference": {
                "type": "float"
            },
            #二階差
            "second order difference": {
                "type": "float"
            },
            #最大心率
            "max heart": {
                "type": "integer"
            },
            #最小心率
            "min heart": {
                "type": "integer"
            },
            #心率變化範圍
            "max to min scope heart": {
                "type": "integer"
            },
            #資訊熵
            "entropy of information": {
                "type": "float"
            },
            #均方根
            "root mean square": {
                "type": "float"
            },
            #標準化均值
            "standardize average": {
                "type": "float"
            }
        }
    }
    return mappings

if __name__ == "__main__":
    es = Elasticsearch(hosts='140.137.41.81', port=9200)
    create_index(es)

