
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from sklearn import tree
import nltk
import copy
stopwordsSet=set() 
for i in ['?',"''","'","'S",'``','.','`',',']:
    stopwordsSet.add(i) # didnt removed "what , when etc" 
with open("train.txt") as f:
    lines=f.readlines()

ngramdict={
    1:500,
    2:300,
    3:200
}

def SplitAndCount(DataLines,ngram):
    data=[]
    Wordslist=[]
    templist=[]
    for i in DataLines:
        sentance=i.strip('\n').split(':')
        sentance[1]=list(sentance[1].lower().split(' '))[1:]
        for j in stopwordsSet:
            if j in sentance[1]:
                sentance[1].remove(j)
        for k in range(len(sentance[1])-(ngram+1)):
            templist.append(' '.join(sentance[1][k:k+ngram]))
        Wordslist+=templist
        templist.clear()
        data+=[[sentance[0],' '.join(sentance[1])]]
    return Wordslist,data 


def count(Wordslist,ngram):
    d={0:1}
    for i in Wordslist:
        d[i]=0
    for i in Wordslist:
        d[i]+=1
    MostFreqNwords=sorted(d.items(),key=lambda x:x[1],reverse=True)[:ngramdict[ngram]]
    MostFreqNwords=[x[0] for x in MostFreqNwords]    
    return MostFreqNwords


def CreateData(ngram,data,MostFreqNwords):
    newdata=copy.deepcopy(data)
    for ind,dat in enumerate(data):
        ofData=list(dat[1].split(' '))
        templist=[]
        for k in range(len(ofData)-(ngram-1)):
            templist.append(' '.join(ofData[k:k+ngram]))
        for i in MostFreqNwords:
            if i in templist:
                newdata[ind].append(1)
            else:
                newdata[ind].append(0)
        newdata[ind].append(len(templist))
    newdf=pd.DataFrame(newdata,columns=['class','sentance',*MostFreqNwords,'length'])

    return newdf

DT=None
featurenames=[]
def start(ErrorCalculationMethod):
    for ngram in range(1,4):
        dataframedict={
                "10 fold round":[],
                "Accuracy"     :[],
                "ngram"        :[],
                "precision"    :[],
                "recall"       :[],
                "F1 score"     :[]
            }
        Wordslist,data=SplitAndCount(lines,ngram)
        MostFreqNwords=count(Wordslist,ngram)
        newdf=CreateData(ngram,data,MostFreqNwords)
        for k in range(10):
            dataframe=copy.deepcopy(newdf)
            totaldata,classes  = dataframe[dataframe.columns[2:]], dataframe['class']
            size=int(len(newdf['class'])/10)
            testdata,testclasses=totaldata.loc[size*k:size*(k+1)],classes.loc[size*k:size*(k+1)]
            traindata=totaldata.drop(list(range(size*k,size*(k+1))),axis=0)
            trainclasses=classes.drop(list(range(size*k,size*(k+1))),axis=0)
            clf = tree.DecisionTreeClassifier(criterion=ErrorCalculationMethod)
            clf = clf.fit(traindata,trainclasses)
            if ngram==1:
                global DT
                DT=clf
                global featurenames
                featurenames=MostFreqNwords
            testresult=clf.predict(testdata)
            scores=precision_recall_fscore_support(testresult, testclasses, average='macro')
            dataframedict["10 fold round"].append(k)
            dataframedict["ngram"].append(ngram)
            dataframedict["Accuracy"].append(list(testresult==testclasses).count(1)/size)
            dataframedict["precision"].append(scores[0])
            dataframedict["recall"].append(scores[1])
            dataframedict['F1 score'].append(scores[2])
        yield dataframedict

for i in ["gini", "entropy", "log_loss"]:
    display(pd.DataFrame([] ,columns=[f"Method={i}"]))
    for dataframe in start(i):
        display(pd.DataFrame(dataframe))
    print("\n\n\n\n\n")

import matplotlib.pyplot as plt
plt.figure(figsize=(100,100))
featurenames+=['somerandom string']
tree.plot_tree(DT,feature_names=featurenames,class_names=[ 'DESC',
                                                            'ENTY',
                                                            'ABBR',
                                                            'HUM',
                                                            'NUM',
                                                            'LOC'  
                                                           ])
plt.savefig('DTree.png', dpi=300)
plt.show()

with open("test.txt") as f:
    testlines=f.readlines()

ngram=1
Wordslist,data=SplitAndCount(testlines,ngram)
testdf=CreateData(ngram,data,featurenames[0:-1])

testdf

TestClassesPredicted=DT.predict(testdf[testdf.columns[2:]])
print("Accuracy",list(TestClassesPredicted==testdf['class']).count(1)/len(testdf['class']))
print(f"\t\t precision score=",
      precision_recall_fscore_support(TestClassesPredicted, testdf['class'], average='macro')[0],
      f"\n\t\t  recall score=",
      precision_recall_fscore_support(TestClassesPredicted, testdf['class'], average='macro')[1],
     f"\n\t\t  F1 score=",
      precision_recall_fscore_support(TestClassesPredicted, testdf['class'], average='macro')[2])


