#!/usr/bin/env python
# coding: utf-8

# In[1]:


from elasticsearch import Elasticsearch
es = Elasticsearch(hosts='140.137.41.81', port=9200)


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


# In[2]:


from elasticsearch import Elasticsearch
import json

#訓練資料
name="yu"
happysubject="電機及資訊科系測量"
sadsubject="設計群測量"

arrsubject=[0.0]*16

for i in range(0,16):
    if i<8:
        def get_queryA():
            queryA = {
                "query": {
                    "bool": {
                        "must": [{
                            "term": {
                                "time Number": i
                            }
                        },
                            {
                            "term": {
                                "name": name
                            }
                        },
                        {
                            "term": {
                                "subject": happysubject
                            }
                        }]
                    }
                }
            }
            return queryA
        if __name__ == "__main__":
            es = Elasticsearch(hosts='140.137.41.81', port=9200)
            queryA = get_queryA()
            resultA = es.search(index='hh3text', body=queryA)
            resulta=json.dumps(resultA, ensure_ascii=False)
            resultPulseA=json.loads(resulta)
            A=resultPulseA['hits']['hits'][0]['_source']['average']
            B=resultPulseA['hits']['hits'][0]['_source']['first order difference']
            C=resultPulseA['hits']['hits'][0]['_source']['second order difference']
            D=resultPulseA['hits']['hits'][0]['_source']['max heart']
            E=resultPulseA['hits']['hits'][0]['_source']['min heart']
            F=resultPulseA['hits']['hits'][0]['_source']['max to min scope heart']
            G=resultPulseA['hits']['hits'][0]['_source']['entropy of information']
            H=resultPulseA['hits']['hits'][0]['_source']['root mean square']
            J=resultPulseA['hits']['hits'][0]['_source']['standardize average']
            arrsubject[i]=[A,B,C,D,E,F,G,H,J]
    else:
        def get_queryB():
            queryB = {
                "query": {
                    "bool": {
                        "must": [{
                            "term": {
                                "time Number": i-8
                            }
                        },
                            {
                            "term": {
                                "name": name
                            }
                        },
                        {
                            "term": {
                                "subject": sadsubject
                            }
                        }]
                    }
                }
            }
            return queryB
        if __name__ == "__main__":
            es = Elasticsearch(hosts='140.137.41.81', port=9200)
            queryB = get_queryB()
            resultB = es.search(index='hh3text', body=queryB)
            resultb=json.dumps(resultB, ensure_ascii=False)
            resultPulseB=json.loads(resultb)
            A=resultPulseB['hits']['hits'][0]['_source']['average']
            B=resultPulseB['hits']['hits'][0]['_source']['first order difference']
            C=resultPulseB['hits']['hits'][0]['_source']['second order difference']
            D=resultPulseB['hits']['hits'][0]['_source']['max heart']
            E=resultPulseB['hits']['hits'][0]['_source']['min heart']
            F=resultPulseB['hits']['hits'][0]['_source']['max to min scope heart']
            G=resultPulseB['hits']['hits'][0]['_source']['entropy of information']
            H=resultPulseB['hits']['hits'][0]['_source']['root mean square']
            J=resultPulseB['hits']['hits'][0]['_source']['standardize average']
            arrsubject[i]=[A,B,C,D,E,F,G,H,J]
print(arrsubject)
y=[1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0]


# In[ ]:




