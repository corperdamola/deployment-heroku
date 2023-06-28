# -*- coding: utf-8 -*-
"""
Created on Tue May 19 12:58:39 2023

@author: Mac PC
"""

import numpy as np # linear algebra 
import pandas as pd # analyze data 
import matplotlib.pyplot as plt 
import seaborn as sns

from google.colab import drive 
drive.mount('/content/drive',force_remount = True)

df=pd.read_csv("/content/drive/MyDrive/pd_speech_features.csv") 
df.head()

def precision(class_id,TP, FP, TN, FN): 
    sonuc=0
for i in range(0,len(class_id)): 
    if (TP[i]==0 or FP[i]==0):
TPen[i]=0.00000001
fpathconf[i]=0.00000001 
sonuc+=(TP[i]/(TP[i]+FP[i]))

sonuc=sonuc/len(class_id) 
return sonuc

def recall(class_id,TP, FP, TN, FN): 
    sonuc=0
for i in range(0,len(class_id)): 
    if (TP[i]==0 or FN[i]==0):
TP[i]=0.00000001
FN[i]=0.00000001 
sonuc+=(TP[i]/(TP[i]+FN[i]))

sonuc=sonuc/len(class_id)
return sonuc

def accuracy(class_id,TP, FP, TN, FN):
sonuc=0
for i in range(0,len(class_id)):
sonuc+=((TP[i]+TN[i])/(TP[i]+FP[i]+TN[i]+FN[i]))

sonuc=sonuc/len(class_id)
return sonuc
def specificity(class_id,TP, FP, TN, FN):
sonuc=0
for i in range(0,len(class_id)):
if (TN[i]==0 or FP[i]==0): 
    TN[i]=0.00000001 
    FP[i]=0.00000001
sonuc+=(TN[i]/(FP[i]+TN[i]))

sonuc=sonuc/len(class_id)
return sonuc
def NPV(class_id,TP, FP, TN, FN):
sonuc=0
for i in range(0,len(class_id)):
if (TN[i]==0 or FN[i]==0): 
    TN[i]=0.00000001 
    FN[i]=0.00000001
sonuc+=(TN[i]/(TN[i]+FN[i]))
sonuc=sonuc/len(class_id)
return sonuc
def perf_measure(y_actual, y_pred):
class_id = set(y_actual).union(set(y_pred)) TP=[]
FP=[]
TN=[]
FN=[]
for index ,_id in enumerate(class_id): 
    TP.append(0)
FP.append(0)
TN.append(0)
FN.append(0)
for i in range(len(y_pred)):
if y_actual[i] == y_pred[i] == _id: 
    TP[index] += 1
if y_pred[i] == _id and y_actual[i] != y_pred[i]: 
    FP[index] += 1
if y_actual[i] == y_pred[i] != _id: TN[index] += 1
if y_pred[i] != _id and y_actual[i] != y_pred[i]: 
    FN[index] += 1
return class_id,TP, FP, TN, FN df.info()
df.columns
man=df.gender.sum() 
total=df.gender.count() 
woman=total-man
print("man: "+str(man)+" woman: "+str(woman)) 
sns.heatmap(df[df.columns[0:10]].corr(),annot=True) 
df.shape
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import f_classif 

y=df["class"]
x=df.iloc[:,2:7]
xnew2=SelectKBest(f_classif, k=5).fit_transform(x, y)
auc_scor=[] 
precision_scor=[] 
x.head()

x=pd.DataFrame(xnew2) 
x.head()

y.value_counts() 
y=y.values type(y)
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=1)

score_liste=[] 
recall_scor=[]
f1_scor=[] 
LR_plus=[] 
LR_eksi=[] 
odd_scor=[] 
NPV_scor=[]
youden_scor=[] 

specificity_scor=[]
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import roc_curve

error_rate = []
for i in range(1,100):
knn = KNeighborsClassifier(n_neighbors=i) 
knn.fit(x_train,y_train)
pred_i = knn.predict(x_test) 
error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6)) 
plt.plot(range(1,100),error_rate,color='blue', linestyle='dashed',
marker='o',markerfacecolor='red', markersize=10) 
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
print("Minimum error:-",min(error_rate),"at K =",error_rate.index(min(error_rate)))
k=10
knn = KNeighborsClassifier(n_neighbors = k) 
knn.fit(x_train,y_train)
y_head=knn.predict(x_test)
print("KNN Algorithm test accuracy",knn.score(x_test,y_test))
classid,tn,fp,fn,tp=perf_measure(y_test,y_head) 
auc_scor.append(roc_auc_score(y_test,y_head)) 
score_list.append(accuracy(classid,tn,fp,fn,tp)) 
precision_scor.append(precision(classid,tn,fp,fn,tp)) 
recall_scor.append(recall(classid,tn,fp,fn,tp)) 
f1_scor.append(f1_score(y_test,y_head,average='macro')) 
NPV_scor.append(NPV(classid,tn,fp,fn,tp)) 
specificity_scor.append(specificity(classid,tn,fp,fn,tp))
LR_plus.append((recall(classid,tn,fp,fn,tp)/(1-specificity(classid,tn,fp,fn,tp)))) 
LR_minus.append(((1-recall(classid,tn,fp,fn,tp))/specificity(classid,tn,fp,fn,tp))) 
odd_scor.append(((recall(classid,tn,fp,fn,tp)/(1-specificity(classid,tn,fp,fn,tp))))/(((1- recall(classid,tn,fp,fn,tp))/specificity(classid,tn,fp,fn,tp)))) 
youden_scor.append((recall(classid,tn,fp,fn,tp)+specificity(classid,tn,fp,fn,tp)-1))
print("KNN algorithm report: \n",classification_report(y_test,y_head))
from sklearn.metrics import confusion_matrix
cmknn = confusion_matrix(y_test,y_head)
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cmknn,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax) 
plt.xlabel("y_pred")
plt.ylabel("y_true") 
plt.title("KNN Algorithm") 
plt.show()
from sklearn.naive_bayes import GaussianNB 
nb=GaussianNB()
nb.fit(x_train,y_train) 
y_head=nb.predict(x_test)
print("Naive Bayes Algorithm test accuracy",nb.score(x_test,y_test))
classid,tn,fp,fn,tp=perf_measure(y_test,y_head) 
auc_scor.append(roc_auc_score(y_test,y_head)) 
score_list.append(accuracy(classid,tn,fp,fn,tp)) 
precision_scor.append(precision(classid,tn,fp,fn,tp)) 
recall_scor.append(recall(classid,tn,fp,fn,tp)) 
f1_scor.append(f1_score(y_test,y_head,average='macro')) 
NPV_scor.append(NPV(classid,tn,fp,fn,tp)) 
specificity_scor.append(specificity(classid,tn,fp,fn,tp))
TPR=recall(classid,tn,fp,fn,tp) 
TNR=specificity(classid,tn,fp,fn,tp)
FPR=1-TNR if FPR==0:
FPR=0.00001 
FNR=1-TPR 
lrminus=FNR/TNR 
lrarti=TPR/FPR
if lrminus==0: 
    lrminus=0.00000001
LR_plus.append(TPR/FPR) 
LR_minus.append(FNR/TNR) 
odd_scor.append(lrarti/lrminus) 
youden_scor.append(TPR+TNR-1)
print("Naive Bayes algorithm report: \n",classification_report(y_test,y_head))
cmnb = confusion_matrix(y_test,y_head)
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cmnb,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax) 
plt.xlabel("y_pred")
plt.ylabel("y_true") 
plt.title("Naive Bayes Algorithm") 
plt.show()
from sklearn.linear_model import LogisticRegression 
lr=LogisticRegression(random_state=0,max_iter=300) 
lr.fit(x_train,y_train)
y_head=lr.predict(x_test)
print("Logistic Regression testaccuracy ",lr.score(x_test,y_test))
classid,tn,fp,fn,tp=perf_measure(y_test,y_head) 
auc_scor.append(roc_auc_score(y_test,y_head)) 
score_list.append(accuracy(classid,tn,fp,fn,tp)) 
precision_scor.append(precision(classid,tn,fp,fn,tp)) 
recall_scor.append(recall(classid,tn,fp,fn,tp)) 
f1_scor.append(f1_score(y_test,y_head,average='macro')) 
NPV_scor.append(NPV(classid,tn,fp,fn,tp)) 
specificity_scor.append(specificity(classid,tn,fp,fn,tp)) 
TPR=recall(classid,tn,fp,fn,tp) 
TNR=specificity(classid,tn,fp,fn,tp)
FPR=1-TNR if FPR==0:
FPR=0.00001 
FNR=1-TPR 
lrminus=FNR/TNR 
lrarti=TPR/FPR
if lrminus==0: lrminus=0.00000001
LR_plus.append(TPR/FPR) 
LR_minus.append(FNR/TNR) 
odd_scor.append(lrarti/lrminus)
youden_scor.append(TPR+TNR-1)
print("Logistic Regression report: \n",classification_report(y_test,y_head))
cmlr = confusion_matrix(y_test,y_head)
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cmlr,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax) 
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Logistic Regression")
plt.show()
algo_list=["KNN","Naive Bayes","Logistic Regression"] 
score={"algo_list":algo_list,"score_list":score_list,"precision":precision_scor,"recall":rec 
       all_scor,"f1_score":f1_scor,"AUC":auc_scor,"LR+":LR_plus,"LR- ":LR_minus,"ODD":odd_scor,"YOUDEN":youden_scor,"Specificity":specificity_scor}
z=pd.DataFrame(score) 
z
f,ax1 = plt.subplots(figsize =(12,12)) 
sns.pointplot(x=df['algo_list'], y=df['score_list'],data=df,color='lime',alpha=0.8,label="score_list") 
sns.pointplot(x=df['algo_list'], y=df['precision'],data=df,color='red',alpha=0.8,label="precision") 
sns.pointplot(x=df['algo_list'], y=df['recall'],data=df,color='black',alpha=0.8,label="recall") 
sns.pointplot(x=df['algo_list'], y=df['f1_score'],data=df,color='blue',alpha=0.8,label="f1_score") 
sns.pointplot(x=df['algo_list'], y=df['AUC'],data=df,color='yellow',alpha=0.8,label="AUC")
sns.pointplot(x=df['algo_list'], y=df['LR- '],data=df,color='orange',alpha=0.8,label="YOUDEN")
sns.pointplot(x=df['algo_list'], y=df['YOUDEN'],data=df,color='brown',alpha=0.8,label="LR-") 
sns.pointplot(x=df['algo_list'], y=df['Specificity'],data=df,color='purple',alpha=0.8,label="Specificity") 
plt.xlabel('Algorithms',fontsize = 15,color='blue')
plt.ylabel('Metrics',fontsize = 15,color='blue')
plt.xticks(rotation= 45)
plt.title('Parkinsons Disease (PD) Evaluation Metrics',fontsize = 20,color='blue') 
plt.grid()
plt.legend()
plt.show()
with open('model.pkl','wb') as f: pickle.dump(nb,f)
model=pickle.load(open('model.pkl','rb')) 
print(model)