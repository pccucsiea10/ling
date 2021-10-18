#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask , jsonify
app = Flask(__name__)
#將ling-張哲唯01text的數據提出來後轉為矩陣



@app.route('/<string:name>/<string:subjecttext>/<int:n>')
def home1(name,subjecttext,n):
    from elasticsearch import Elasticsearch
    import json
    
    from sklearn.neighbors import KNeighborsClassifier
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    import operator
    import math
    
    #訓練資料
    happysubject="電機及資訊科系測量"

    arrX0=[0.0]*16
    arrX1=[0.0]*16
    arrX2=[0.0]*16
    arrX3=[0.0]*16
    arrX4=[0.0]*16
    arrX5=[0.0]*16
    arrX6=[0.0]*16
    arrX7=[0.0]*16
    arrX8=[0.0]*16

    for i in range(0,16):
        if i<8:
            def get_queryA():
                queryA = {
                    "query": {
                        "bool": {
                            "must": [{
                                "term": {
                                    "Number": i
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
                arrX0[i]=A
                B=resultPulseA['hits']['hits'][0]['_source']['first order difference']
                arrX1[i]=B
                C=resultPulseA['hits']['hits'][0]['_source']['second order difference']
                arrX2[i]=C
                D=resultPulseA['hits']['hits'][0]['_source']['max heart']
                arrX3[i]=D
                E=resultPulseA['hits']['hits'][0]['_source']['min heart']
                arrX4[i]=E
                F=resultPulseA['hits']['hits'][0]['_source']['max to min scope heart']
                arrX5[i]=F
                G=resultPulseA['hits']['hits'][0]['_source']['entropy of information']
                arrX6[i]=G
                H=resultPulseA['hits']['hits'][0]['_source']['root mean square']
                arrX7[i]=H
                J=resultPulseA['hits']['hits'][0]['_source']['standardize average']
                arrX8[i]=J
        else:
            def get_queryB():
                queryB = {
                    "query": {
                        "bool": {
                            "must": [{
                                "term": {
                                    "Number": i-8
                                }
                            },
                                {
                                "term": {
                                    "name": name
                                }
                            },
                            {
                                "term": {
                                    "subject": subjecttext
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
                arrX0[i]=A
                B=resultPulseB['hits']['hits'][0]['_source']['first order difference']
                arrX1[i]=B
                C=resultPulseB['hits']['hits'][0]['_source']['second order difference']
                arrX2[i]=C
                D=resultPulseB['hits']['hits'][0]['_source']['max heart']
                arrX3[i]=D
                E=resultPulseB['hits']['hits'][0]['_source']['min heart']
                arrX4[i]=E
                F=resultPulseB['hits']['hits'][0]['_source']['max to min scope heart']
                arrX5[i]=F
                G=resultPulseB['hits']['hits'][0]['_source']['entropy of information']
                arrX6[i]=G
                H=resultPulseB['hits']['hits'][0]['_source']['root mean square']
                arrX7[i]=H
                J=resultPulseB['hits']['hits'][0]['_source']['standardize average']
                arrX8[i]=J
    print("arrX0=",arrX0)
    print("arrX1=",arrX1)
    print("arrX2=",arrX2)
    print("arrX3=",arrX3)
    print("arrX4=",arrX4)
    print("arrX5=",arrX5)
    print("arrX6=",arrX6)
    print("arrX7=",arrX7)
    print("arrX8=",arrX8)
    y=[1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0]

    #計算準確率
    arra = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    number = '{:09b}'.format(n, "b")
    for p in range(0,9):
        arra[p]=int(number[p])
        print(arra[p],end='')
    test = []
    if arra[0] == 1:
        if len(test):
            test = test
        else:
            test = np.c_[arrX0]
    if arra[1] == 1:
        if len(test):
            test = np.c_[test, arrX1]
        else:
            test = np.c_[arrX1]
    if arra[2] == 1:
        if len(test):
            test = np.c_[test, arrX2]
        else:
            test = np.c_[arrX2]
    if arra[3] == 1:
        if len(test):
            test = np.c_[test, arrX3]
        else:
            test = np.c_[arrX3]
    if arra[4] == 1:
        if len(test):
            test = np.c_[test, arrX4]
        else:
            test = np.c_[arrX4]
    if arra[5] == 1:
        if len(test):
            test = np.c_[test, arrX5]
        else:
            test = np.c_[arrX5]
    if arra[6] == 1:
        if len(test):
            test = np.c_[test,arrX6]
        else:
            test = np.c_[arrX6]
    if arra[7] == 1:
        if len(test):
            test = np.c_[test, arrX7]
        else:
            test = np.c_[arrX7]
    if arra[8] == 1:
        if len(test):
            test = np.c_[test, arrX8]
        else:
            test = np.c_[arrX8]

    arra[8] += 1
    X_train, X_test, y_train, y_test = train_test_split(test, y,test_size=0.3,random_state=1)
    maxScore = 0
    kn = 0
    for o in range(1, 8):
        clf = KNeighborsClassifier(n_neighbors = o,p = 2,weights = 'distance',algorithm = 'auto')
        clf.fit(X_train,y_train)
        score = clf.score(X_test,y_test)
        if score > maxScore:
            maxScore = score
            kn = o
    print()
    print(maxScore,kn)
    return jsonify(maxScore)
if __name__ == "__main__":
    app.run()


# In[ ]:




