#!/usr/bin/env python
# coding: utf-8

# In[8]:


from flask import Flask , jsonify
app = Flask(__name__)


@app.route('/<string:name>/<string:trial>/<int:n>')
def home1(name,trial,n):
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
    
    arra = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    number = '{:09b}'.format(n, "b")
    for p in range(0,9):
        arra[p]=int(number[p])
    
    for i in range(0,3):
        if i==0:
            happysubject="第二次測量"
            subjectbasic=happysubject
            arrif=[74,75,73,74,76,75,74,76,78,84]
        elif i==1:
            sadsubject= "第一次測量"
            subjectbasic=sadsubject
            arrif=[80,79,78,80,81,82,80,81,80,81]
        else :
            testsubject= trial
            subjectbasic=testsubject
            arrif=[90,91,89,79,76,77,75,74,76,81]
        def get_querybasic():
            querybasic = {
                "query": {
                    "bool": {
                        "must": [
                            {
                            "term": {
                                "name": name
                            }
                        },
                            {
                            "term": {
                                "subject.keyword": {"value": subjectbasic}
                            }
                        }]
                    }
                }
            }
            return querybasic
        if __name__ == "__main__":
            es = Elasticsearch(hosts='140.137.41.81', port=9200)
            querybasic = get_querybasic()
            resultbasic = es.search(index='hh3', body=querybasic)
            resultbasic1=json.dumps(resultbasic, ensure_ascii=False)
            resultPulsebasic=json.loads(resultbasic1)
            size=resultPulsebasic['hits']['total']['value']
            if size>10:
                size = 10
            arrbasic=[0]*size
            arrbasic1=[0]*size
            arrbasic2=[0]*10
            for a in range(0,size):
                pulse=(resultPulsebasic['hits']['hits'][a]['_source']['pulse'])
                time=(resultPulsebasic['hits']['hits'][a]['_source']['timeForCompare'])        
                arrbasic[a]=pulse,time
            arrbasic1=sorted(arrbasic,key=lambda s:s[1])
            for b in range(0,size):
                arrbasic2[b]=arrbasic1[b][0]
            for c in range(size,10):
                arrbasic2[c]=arrif[c]
            if i==0:
                arrhappy2=arrbasic2
            elif i==1:
                arrsad2=arrbasic2
            elif i==2:
                arrtest2=arrbasic2
    
    arrhappytext1=[0]*8
    arrsadtext1=[0]*8
    arrtesttext1=[0]*8
    for i in range (0,8):
        arrhappytext1[i]=[arrhappy2[i],arrhappy2[i+1],arrhappy2[i+2]]
        arrsadtext1[i]=[arrsad2[i],arrsad2[i+1],arrsad2[i+2]]
        arrtesttext1[i]=[arrtest2[i],arrtest2[i+1],arrtest2[i+2]]
    arrhappycount=[0.0]*8
    arrsadcount=[0.0]*8
    arrtestcount=[0.0]*8
    for z in range(0,8):
        happysize=3
        sadsize=3
        testsize=3

        #平均
        aveB = 0.0
        for i in range(0,happysize):
            aveB +=arrhappytext1[z][i]
        aveB = aveB/happysize

        #DiffB1 一階差的平均值
        diffB1 = 0.0
        meanB1 = 0.0
        #X[n-1]-X[n]的總和
        for i in range(0,(happysize-1)):
            meanB1+=abs(arrhappytext1[z][i+1]-arrhappytext1[z][i])
        diffB1 = meanB1/(happysize-1)

        #DiffB2 二階差的平均值
        diffB2 = 0.0
        meanB2 = 0.0
        #X[n+2]-X[n]的總和
        for i in range(0,(happysize-2)):
            meanB2 += abs(arrhappytext1[z][i+2]-arrhappytext1[z][i])
        diffB2 = meanB2/(happysize-2)

        #range 心率變化範圍
        scopeB = 0
        scopemaxB = 0
        scopeminB = 110

        for i in range(0,happysize):
            #求最大心率
            if arrhappytext1[z][i]>scopemaxB:
                scopemaxB = arrhappytext1[z][i]
            if arrhappytext1[z][i]<scopeminB:
                scopeminB = arrhappytext1[z][i]
        scopeB = scopemaxB-scopeminB

        #H(X)資訊熵
        x = 0
        HB = 0
        arrpb = [0.0]*happysize
        for i in range(0,happysize):
            x=0 #初值為0
            #心率出現次數
            for j in range(0,happysize):
                if arrhappytext1[z][i]==arrhappytext1[z][j]:
                    x+=1
            #p[i]機率
            arrpb[i] = x*1.0/happysize
        for k in range(0,happysize):
            HB += -arrpb[k]*math.log(arrpb[k],2)

        #Radrm 均方根
        radrmB = 0.0   
        meanrB = 0.0
        #相鄰值平方差的總和
        for i in range(0,(happysize-1)):
            radrmB += pow(arrhappytext1[z][i+1]-arrhappytext1[z][i],2)
        meanrB = radrmB/happysize

        arrhappycount[z]=aveB,diffB1,diffB2,scopemaxB,scopeminB,scopeB,HB,meanrB
        def load_datas():
            datas = list()
            datas.append(
            {
                #匯入編號
                "time Number": z,
                #匯入名字
                "name": name,
                #匯入影片類別
                "subject": happysubject,
                #匯入平均
                "average": float(aveB),
                #匯入一階差
                "first order difference":float(diffB1),
                #匯入二階差
                "second order difference":float(diffB2),
                #匯入最大心率
                "max heart":int(scopemaxB),
                #匯入最小心率
                "min heart":int(scopeminB),
                #匯入心率變化範圍
                "max to min scope heart":int(scopeB),
                #匯入資訊熵
                "entropy of information":float(HB),
                #匯入均方根
                "root mean square":float(meanrB)
            }
            )
            return datas

        def create_data(es, datas):
            for data in datas:
                es.index(index='hh3text', body=data)

        if __name__ == "__main__":
            es = Elasticsearch(hosts='140.137.41.81', port=9200)
            datas = load_datas()
            create_data(es, datas)

        #平均
        aveD = 0.0
        for i in range(0,sadsize):
            aveD +=arrsadtext1[z][i]
        aveD = aveD/sadsize

        #DiffD1 一階差的平均值
        diffD1 = 0.0
        meanD1 = 0.0
        #X[n+1]-X[n]的總和
        for i in range(0,(sadsize-1)):
                meanD1 += abs(arrsadtext1[z][i+1]-arrsadtext1[z][i])
        diffD1 = meanD1/(sadsize-1)

        #DiffD2 二階差的平均值
        diffD2 = 0.0
        meanD2 = 0.0
        #X[n+2]-X[n]的總和
        for i in range(0,(sadsize-2)):
            meanD2 += abs(arrsadtext1[z][i+2]-arrsadtext1[z][i])
        diffD2 = meanD2/(sadsize-2)

        #range 心率變化範圍
        scopeD = 0
        scopemaxD = 0
        scopeminD = 110
        for i in range(0,sadsize):
            #求最大心率
            if arrsadtext1[z][i]>scopemaxD:
                scopemaxD = arrsadtext1[z][i]
            #求最小心率
            if arrsadtext1[z][i]<scopeminD:
                scopeminD = arrsadtext1[z][i]
        scopeD = scopemaxD-scopeminD

        #H(X)資訊熵
        x = 0
        HD = 0
        arrpd = [0.0]*sadsize
        for i in range(0,sadsize):
            x = 0 #初值為0 
            #心率出現次數
            for j in range(0,sadsize):
                if arrsadtext1[z][i]==arrsadtext1[z][j]:
                    x+=1
            arrpd[i] = x*1.0/sadsize
        for k in range(0,sadsize):
            HD += -arrpd[k]*math.log(arrpd[k],2)

        #Radrm 均方根
        radrmD = 0.0
        meanrD = 0.0
        #相鄰值平方差的總和
        for i in range(0,(sadsize-1)):
            radrmD += pow(arrsadtext1[z][i+1]-arrsadtext1[z][i],2)
        meanrD = radrmD/sadsize

        arrsadcount[z]=aveD,diffD1,diffD2,scopemaxD,scopeminD,scopeD,HD,meanrD
        
        #匯入資料
        def load_datas():
            datas = list()
            datas.append(
            {
                #匯入次序編號
                "time Number": z,
                #匯入名字
                "name": name,
                #匯入影片類別
                "subject": sadsubject,
                #匯入平均
                "average": float(aveD),
                #匯入一階差
                "first order difference":float(diffD1),
                #匯入二階差
                "second order difference":float(diffD2),
                #匯入最大心率
                "max heart":int(scopemaxD),
                #匯入最小心率
                "min heart":int(scopeminD),
                #匯入心率變化範圍
                "max to min scope heart":int(scopeD),
                #匯入資訊熵
                "entropy of information":float(HD),
                #匯入均方根
                "root mean square":float(meanrD)
            }
            )
            return datas

        def create_data(es, datas):
            for data in datas:
                es.index(index='hh3text', body=data)

        if __name__ == "__main__":
            es = Elasticsearch(hosts='140.137.41.81', port=9200)
            datas = load_datas()
            create_data(es, datas)
            
        #平均
        aveE = 0.0
        for i in range(0,testsize):
            aveE +=arrtesttext1[z][i]
        aveE = aveE/testsize

        #DiffE1 一階差的平均值
        diffE1 = 0.0
        meanE1 = 0.0
        #X[n-1]-X[n]的總和
        for i in range(0,(testsize-1)):
            meanE1+=abs(arrtesttext1[z][i+1]-arrtesttext1[z][i])
        diffE1 = meanE1/(testsize-1)

        #DiffE2 二階差的平均值
        diffE2 = 0.0
        meanE2 = 0.0
        #X[n+2]-X[n]的總和
        for i in range(0,(testsize-2)):
            meanE2 += abs(arrtesttext1[z][i+2]-arrtesttext1[z][i])
        diffE2 = meanE2/(testsize-2)

        #range 心率變化範圍
        scopeE = 0
        scopemaxE = 0
        scopeminE = 110

        for i in range(0,testsize):
            #求最大心率
            if arrtesttext1[z][i]>scopemaxE:
                scopemaxE = arrtesttext1[z][i]
            if arrtesttext1[z][i]<scopeminE:
                scopeminE = arrtesttext1[z][i]
        scopeE = scopemaxE-scopeminE

        #H(X)資訊熵
        x = 0
        HE = 0
        arrpb = [0.0]*testsize
        for i in range(0,testsize):
            x=0 #初值為0
            #心率出現次數
            for j in range(0,testsize):
                if arrtesttext1[z][i]==arrtesttext1[z][j]:
                    x+=1
            #p[i]機率
            arrpb[i] = x*1.0/testsize
        for k in range(0,testsize):
            HE += -arrpb[k]*math.log(arrpb[k],2)

        #Radrm 均方根
        radrmE = 0.0   
        meanrE = 0.0
        #相鄰值平方差的總和
        for i in range(0,(testsize-1)):
            radrmE += pow(arrtesttext1[z][i+1]-arrtesttext1[z][i],2)
        meanrE = radrmE/testsize

        arrtestcount[z]=aveE,diffE1,diffE2,scopemaxE,scopeminE,scopeE,HE,meanrE
        def load_datas():
            datas = list()
            datas.append(
            {
                #匯入編號
                "time Number": z,
                #匯入名字
                "name": name,
                #匯入影片類別
                "subject": testsubject,
                #匯入平均
                "average": float(aveE),
                #匯入一階差
                "first order difference":float(diffE1),
                #匯入二階差
                "second order difference":float(diffE2),
                #匯入最大心率
                "max heart":int(scopemaxE),
                #匯入最小心率
                "min heart":int(scopeminE),
                #匯入心率變化範圍
                "max to min scope heart":int(scopeE),
                #匯入資訊熵
                "entropy of information":float(HE),
                #匯入均方根
                "root mean square":float(meanrE)
            }
            )
            return datas

        def create_data(es, datas):
            for data in datas:
                es.index(index='hh3text', body=data)

        if __name__ == "__main__":
            es = Elasticsearch(hosts='140.137.41.81', port=9200)
            datas = load_datas()
            create_data(es, datas)

    arrsubject=[0.0]*16

    for i in range(0,16):
        if i<8:
            arrsubject[i]=arrhappycount[i]
        else:
            arrsubject[i]=arrsadcount[i-8]

    y=[1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0]

    import numpy as np 
    from sklearn.feature_selection import SelectKBest, chi2
    x = np.array(arrsubject)

    
    kbest = SelectKBest(chi2, k=3)

    #特徵選取
    X_new = kbest.fit_transform(x, y)

    ##通過KNN進行分類
    def classify(input, X, y):
        global classes
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
    arrresult=[0.0]*8
    selected = []
    if n == 888:
        selected = kbest.get_support()
    else:
        X_n = []
        for j in range(0,8):          
            if arra[j] == 1:
                selected.append(True)
            else:
                selected.append(False)  
        for j in range(0,16):
            s = []
            for k in range(0,8):          
                if arra[k] == 1:
                     s.append(x[j][k])
            X_n.append(s)
        X_new = np.array(X_n)           
    for i in range(0, 8):
        input = []
        for j in range(0,8):          
            if selected[j] == True:
                input.append(arrtestcount[i][j])
        arrresult[i]=classify(input, X_new, y)
    r=sum(arrresult)/8*100
    d={}
    d["result"]=r
    return jsonify(d)
if __name__ == "__main__":
    app.run()


# In[ ]:




