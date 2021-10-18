#!/usr/bin/env python
# coding: utf-8

# In[4]:


#!/usr/bin/env python
# coding: utf-8

# In[20]:


from elasticsearch import Elasticsearch
es = Elasticsearch(hosts='140.137.41.81', port=9200)

from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
import operator
import math
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
            arrsubject[i]=[A,B,C,D,E,F,G,H]
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
            arrsubject[i]=[A,B,C,D,E,F,G,H]
print(arrsubject)
y=[1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0]

import numpy as np 
from sklearn.feature_selection import SelectKBest, chi2
x = np.array(arrsubject)
print('原本x的維度=(數據, 特徵) : ', x.shape) 

kbest = SelectKBest(chi2, k=3)
#特徵選取
X_new = kbest.fit_transform(x, y)
print('訓練後x的維度=(數據, 特徵): ', X_new.shape)
print('原本x數據: \n', x)
print('訓練後x數據: \n', X_new)
print('訓練後x數據(含去掉的特徵): \n', kbest.inverse_transform(X_new))
print('KBest引數=(k值, 演算法): ', kbest.get_params())
print('使用的特徵: ', kbest.get_support())
print('查詢特徵卡方值(越大越好): ', kbest.scores_)
print('查詢pvalue: ', kbest.pvalues_)

##通過KNN進行分類
def classify(input, X, y):
    dataSize = X.shape[0]
    ## 重複input為dataSet的大小
    diff = np.tile(input, (dataSize, 1)) - X
    sqdiff = diff**2
    ## 列向量分別相加，得到一列新的向量
    squareDist = np.array([sum(z) for z in sqdiff])
    dist = squareDist**0.5
    
    ## 對距離進行排序
    ## argsort()根據元素的值從大到小對元素進行排序，返回下標
    sortedDistIndex = np.argsort(dist)

    # 給距離加入權重
    w = []
    for i in range(5):
        w.append((dist[sortedDistIndex[5-1]] - dist[sortedDistIndex[i]])                 / (dist[sortedDistIndex[5-1]] - dist[sortedDistIndex[0]]))
    
    classCount = {}
    temp = 0
    for i in range(3):
        ## 因為已經對距離進行排序，所以直接迴圈sortedDistIndx
        voteLabel = y[sortedDistIndex[i]]
        ## 對選取的k個樣本所屬的類別個數進行統計
        ## 如果獲取的標籤不在classCount中，返回0
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1 + w[temp]
        temp += 1
    ## 選取出現的類別次數最多的類別
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            classes = key
    
    return classes

for i in range(0, 8):
    def get_queryC():
            queryC = {
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
            return queryC
    if __name__ == "__main__":
        es = Elasticsearch(hosts='140.137.41.81', port=9200)
        queryC = get_queryC()
        resultC = es.search(index='hh3text', body=queryC)
        resultc=json.dumps(resultC, ensure_ascii=False)
        resultPulseC=json.loads(resultc)
        A=resultPulseC['hits']['hits'][0]['_source']['average']
        B=resultPulseC['hits']['hits'][0]['_source']['max heart']
        C=resultPulseC['hits']['hits'][0]['_source']['root mean square']
        input=[A,D,H]
    print(classify(input, X_new, y))
print()


# In[ ]:




