#!/usr/bin/env python
# coding: utf-8

# In[34]:


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
    for i in range(0,4):
        if i==0:
            naturalsubject="正常心率測量"
            subjectbasic=naturalsubject
            arrif=[85,86,85,78,77,75,74,73,72,74]
        elif i==1:
            happysubject="第二次測量"
            subjectbasic=happysubject
            arrif=[74,75,73,74,76,75,74,76,78,84]
        elif i==2:
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
                arrbasic=[0]*10
                arrbasic1=[0]*10
                arrbasic2=[0]*10
                for a in range(0,10):
                    pulse=(resultPulsebasic['hits']['hits'][a]['_source']['pulse'])
                    time=(resultPulsebasic['hits']['hits'][a]['_source']['timeForCompare'])        
                    arrbasic[a]=pulse,time
                arrbasic1=sorted(arrbasic,key=lambda s:s[1])
                for b in range(0,10):
                    arrbasic2[b]=arrbasic1[b][0]
                if i==0:
                    arrnatural2=arrbasic2
                elif i==1:
                    arrhappy2=arrbasic2
                elif i==2:
                    arrsad2=arrbasic2
                else:
                    arrtest2=arrbasic2
            else:
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
                    arrnatural2=arrbasic2
                elif i==1:
                    arrhappy2=arrbasic2
                elif i==2:
                    arrsad2=arrbasic2
                elif i==3:
                    arrtest2=arrbasic2
    print(arrnatural2)
    print(arrhappy2)
    print(arrsad2)
    print(arrtest2)
    
    arrnaturaltext1=[0]*8
    arrhappytext1=[0]*8
    arrsadtext1=[0]*8
    arrtesttext1=[0]*8
    for i in range (0,8):
        arrnaturaltext1[i]=[arrnatural2[i],arrnatural2[i+1],arrnatural2[i+2]]
        arrhappytext1[i]=[arrhappy2[i],arrhappy2[i+1],arrhappy2[i+2]]
        arrsadtext1[i]=[arrsad2[i],arrsad2[i+1],arrsad2[i+2]]
        arrtesttext1[i]=[arrtest2[i],arrtest2[i+1],arrtest2[i+2]]
    print(arrnaturaltext1)
    print(arrhappytext1)
    print(arrsadtext1)
    print(arrtesttext1)
    arrhappycount=[0.0]*8
    arrsadcount=[0.0]*8
    arrtestcount=[0.0]*8
    for z in range(0,8):
        print(z)
        naturalsize=3
        happysize=3
        sadsize=3
        testsize=3
        #正常心率平均
        ave = 0.0
        for i in range(0,naturalsize):
            ave+=arrnaturaltext1[z][i]
        ave = ave/naturalsize

        print("happy心率特徵 :")

        #平均
        aveB = 0.0
        for i in range(0,happysize):
            aveB +=arrhappytext1[z][i]
        aveB = aveB/happysize
        print("平均: ",aveB)

        #DiffB1 一階差的平均值
        diffB1 = 0.0
        meanB1 = 0.0
        #X[n-1]-X[n]的總和
        for i in range(0,(happysize-1)):
            meanB1+=abs(arrhappytext1[z][i+1]-arrhappytext1[z][i])
        diffB1 = meanB1/(happysize-1)
        print("一階差: ",diffB1)

        #DiffB2 二階差的平均值
        diffB2 = 0.0
        meanB2 = 0.0
        #X[n+2]-X[n]的總和
        for i in range(0,(happysize-2)):
            meanB2 += abs(arrhappytext1[z][i+2]-arrhappytext1[z][i])
        diffB2 = meanB2/(happysize-2)
        print("二階差: ",diffB2)

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
        print("最大心率: ",scopemaxB)
        print("最小心率: ",scopeminB)
        print("心率變化範圍: ",scopeB)

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
        print("資訊熵: ",HB)

        #Radrm 均方根
        radrmB = 0.0   
        meanrB = 0.0
        #相鄰值平方差的總和
        for i in range(0,(happysize-1)):
            radrmB += pow(arrhappytext1[z][i+1]-arrhappytext1[z][i],2)
        meanrB = radrmB/happysize
        print("均方根: ",meanrB)

        #標準化均值
        arrbb=[0.0]*happysize
        aveb = 0.0
        for i in range(0,happysize):
            arrbb[i] = arrhappytext1[z][i]-ave
            aveb += arrbb[i]
        aveb = aveb/happysize
        print("標準化均值: ",aveb)
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
                "root mean square":float(meanrB),
                #匯入標準化均值
                "standardize average":float(aveb),
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
        
        print("sad心率特徵: ")

        #平均
        aveD = 0.0
        for i in range(0,sadsize):
            aveD +=arrsadtext1[z][i]
        aveD = aveD/sadsize
        print("平均: ",aveD)

        #DiffD1 一階差的平均值
        diffD1 = 0.0
        meanD1 = 0.0
        #X[n+1]-X[n]的總和
        for i in range(0,(sadsize-1)):
                meanD1 += abs(arrsadtext1[z][i+1]-arrsadtext1[z][i])
        diffD1 = meanD1/(sadsize-1)
        print("一階差: ",diffD1)

        #DiffD2 二階差的平均值
        diffD2 = 0.0
        meanD2 = 0.0
        #X[n+2]-X[n]的總和
        for i in range(0,(sadsize-2)):
            meanD2 += abs(arrsadtext1[z][i+2]-arrsadtext1[z][i])
        diffD2 = meanD2/(sadsize-2)
        print("二階差: ",diffD2)

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
        print("最大心率: ",scopemaxD)
        print("最小心率: ",scopeminD)
        print("心率變化範圍: ",scopeD)

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
        print("資訊熵: ",HD)

        #Radrm 均方根
        radrmD = 0.0
        meanrD = 0.0
        #相鄰值平方差的總和
        for i in range(0,(sadsize-1)):
            radrmD += pow(arrsadtext1[z][i+1]-arrsadtext1[z][i],2)
        meanrD = radrmD/sadsize
        print("均方根: ",meanrD)

        #標準化均值
        arrdd = [0.0]*sadsize
        aved = 0.0
        for i in range(0,sadsize):
            arrdd[i] = arrsadtext1[z][i] -ave
            aved +=arrdd[i]
        aved = aved/sadsize
        print("標準化均值: ",aved)
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
                "root mean square":float(meanrD),
                #匯入標準化均值
                "standardize average":float(aved),
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
            
        print("subject心率特徵: ")
        #平均
        aveE = 0.0
        for i in range(0,testsize):
            aveE +=arrtesttext1[z][i]
        aveE = aveE/testsize
        print("平均: ",aveE)

        #DiffE1 一階差的平均值
        diffE1 = 0.0
        meanE1 = 0.0
        #X[n-1]-X[n]的總和
        for i in range(0,(testsize-1)):
            meanE1+=abs(arrtesttext1[z][i+1]-arrtesttext1[z][i])
        diffE1 = meanE1/(testsize-1)
        print("一階差: ",diffE1)

        #DiffE2 二階差的平均值
        diffE2 = 0.0
        meanE2 = 0.0
        #X[n+2]-X[n]的總和
        for i in range(0,(testsize-2)):
            meanE2 += abs(arrtesttext1[z][i+2]-arrtesttext1[z][i])
        diffE2 = meanE2/(testsize-2)
        print("二階差: ",diffE2)

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
        print("最大心率: ",scopemaxE)
        print("最小心率: ",scopeminE)
        print("心率變化範圍: ",scopeE)

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
        print("資訊熵: ",HE)

        #Radrm 均方根
        radrmE = 0.0   
        meanrE = 0.0
        #相鄰值平方差的總和
        for i in range(0,(testsize-1)):
            radrmE += pow(arrtesttext1[z][i+1]-arrtesttext1[z][i],2)
        meanrE = radrmE/testsize
        print("均方根: ",meanrE)

        #標準化均值
        arree=[0.0]*testsize
        avee = 0.0
        for i in range(0,testsize):
            arree[i] = arrtesttext1[z][i]-ave
            avee += arree[i]
        avee = avee/testsize
        print("標準化均值: ",avee)
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
                "root mean square":float(meanrE),
                #匯入標準化均值
                "standardize average":float(avee),
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
    print(arrhappycount)
    print(arrsadcount)
    print(arrtestcount)

    arrsubject=[0.0]*16

    for i in range(0,16):
        if i<8:
            arrsubject[i]=arrhappycount[i]
        else:
            arrsubject[i]=arrsadcount[i-8]
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
    for i in range(0, 8):
        input=[arrtestcount[i][2],arrtestcount[i][5],arrtestcount[i][7]]
        print(classify(input, X_new, y))
        arrresult[i]=classify(input, X_new, y)
    return jsonify(arrresult)
if __name__ == "__main__":
    app.run()


# In[ ]:




