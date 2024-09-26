# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 13:09:18 2018

@author: nitr
"""

import pandas as pd
import timeit
import xlsxwriter
from  sklearn.preprocessing import LabelEncoder
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from datetime import datetime

import fun_sacling_function as sff
ScalingMethods=sff.scaling_fun()
import fun_Classification_Function as cff
models=cff.class_fun()
import glob
path = r'D:\Krishnasai\2@ResearchRelated\Existing_Work_With_My_Implementation\1sec_600EP_ExistingWorkResults\S_EEG_2014_1\Oz_Driver_1Sec_600Epochs' 
l=glob.glob(path + "/*.csv")
print('Total No of csv files',len(l))
import numpy as np
data = np.asarray(l)
ff=l[0];
sff=ff[-48:];
sff=sff[:-17];
Fre='100';
print(sff)
labe="CombineSub_"+sff+"_"+"(19f)";
ftacu=path+'\\'+labe+'_TotalAacuracy.xlsx';
fSen=path+'\\'+labe+'_Sensitivity.xlsx';
fPrec=path+'\\'+labe+'_Precision.xlsx';
ff1S=path+'\\'+labe+'_f1SCORE.xlsx';
# =============================================================================
    # Creating excel file for storing results
workbook = xlsxwriter.Workbook(ftacu)
worksheet = workbook.add_worksheet('TotalAacuracy')

workbook1 = xlsxwriter.Workbook(fSen)
worksheet1 = workbook1.add_worksheet('Sensitivity')

workbook2 = xlsxwriter.Workbook(fPrec)
worksheet2 = workbook2.add_worksheet('Precision')

workbook3 = xlsxwriter.Workbook(ff1S)
worksheet3 = workbook3.add_worksheet('f1SCOR')

rowx=0;
colx=0;
cnam=['Sno','Name','Active','Drowsy','','Classificatoin'];
for ScNam, Scmethod  in ScalingMethods:
    cnam.append(ScNam)
worksheet.write_row(rowx,colx,cnam)
worksheet1.write_row(rowx,colx,cnam)
worksheet2.write_row(rowx,colx,cnam)
worksheet3.write_row(rowx,colx,cnam)

Prog_start_time = timeit.default_timer();
Prog_st_time=datetime.now().strftime('%d-%m-%Y %H:%M:%S');
for f in range(len(l)):
    mydataset=pd.read_csv(l[f])
    print(f,':-',l[f],'Samples= ',mydataset.shape[0], end='')
    if f==0:
        df1=mydataset.copy()
        continue
    df1 = df1.append([mydataset])
    print(' Till The Sample ',df1.shape[0])

mydataset = df1
import numpy
# mark zero values as missing or NaN
dataset= mydataset.replace(0, numpy.NaN)
# fill missing values with mean column values
dataset.fillna(dataset.mean(), inplace=True)

# mark zero values as missing or inf
dataset= mydataset.replace(0, numpy.inf)
# fill missing values with mean column values
dataset.fillna(dataset.mean(), inplace=True)


mydataset=dataset;



from sklearn.utils import shuffle
df = shuffle(mydataset)
mydataset=df;


i=1;
for name, clf  in models:
    [rc, cc]=mydataset.shape;
    #array = dataframe.values
    MainMyX = mydataset.iloc[:,:-1].values
    MyY=mydataset.iloc[:,cc-1].values
    W=pd.DataFrame(MyY) # to convert from object to data frame
    # converstion of stiring class label (A and B) to integer (0 and 1)
    # from  sklearn.preprocessing import LabelEncoder
    labelencoder_X=LabelEncoder()
    MyY=labelencoder_X.fit_transform(MyY)
    MyYY=labelencoder_X.fit_transform(MyY)
    from collections import Counter
    values=Counter(MyY)
    list2=[i,'',values[0],values[1],'']
    list2=[i,'',values[0],values[1],'',name]
    l1_list2=[i,'',values[0],values[1],'',name]
    l2_list2=[i,'',values[0],values[1],'',name]
    l3_list2=[i,'',values[0],values[1],'',name]
# =============================================================================
#     np.all(np.isfinite(MainMyX))
#     print((MainMyX[[5]] == 0).sum())
# =============================================================================
   
    p=MainMyX;
    for j in range(19):
        index = 0
        for i in p[:,j]:
            if not np.isfinite(i):
                print("feture",j,'col',index,'  ', i)
                MainMyX[index,j]=(MainMyX[index-1,0]+MainMyX[index-2,0])/2;
            index +=1



    ''' Model Selectin'''   
    for ScNam, Scmethod  in ScalingMethods:
        #-----------------------------
        ''' Scaling'''
        if ScNam is 'WithoutScaling':
            MyX=MainMyX
        else:
            MyX=Scmethod.fit_transform(MainMyX) 
        #-----------------------------
        print (i,':-','Class:',name,' scale:',ScNam,end='')
        MyY=MyYY;
        if name is 'ANN':
            MyY=MyYY;
            X_train, X_test, y_train, y_test = train_test_split(MyX,MyYY, test_size=0.30, random_state=42)
            clf.fit(X_train, y_train, nb_epoch=100, batch_size=5)
            y_pred = clf.predict_classes(X_test);
            cc=y_test.reshape(y_test.size,1);
            MyY=cc;
        else:
            MyY=MyYY
            y_pred = cross_val_predict(clf,MyX,MyY,cv=10)

        totacu=round((metrics.accuracy_score(MyY,y_pred)*100),3)
        totMisacu=round((1-metrics.accuracy_score(MyY,y_pred))*100,3)
        sensitivityVal=round((metrics.recall_score(MyY,y_pred))*100,3)
        precision=round((metrics.precision_score(MyY,y_pred))*100,3);
        f1score=round(2*((sensitivityVal*precision)/(sensitivityVal+precision)),3)
        print ('tA(',totacu,') tM(',totMisacu,') Se(',sensitivityVal,') pe(',precision,') F1(',f1score,')')
        import math
        if math.isnan(totacu):
            totacu=0;
        if math.isnan(totMisacu):
            totMisacu=0;
        if math.isnan(sensitivityVal):
            sensitivityVal=0;
        if math.isnan(precision):
            precision=0;
        if math.isnan(f1score):
            f1score=0;
        list2.append(totacu)
        l1_list2.append(sensitivityVal)
        l2_list2.append(precision)
        l3_list2.append(f1score)
    rowx=rowx+1;
    worksheet.write_row(rowx,colx,list2)
    worksheet1.write_row(rowx,colx,l1_list2)
    worksheet2.write_row(rowx,colx,l2_list2)
    worksheet3.write_row(rowx,colx,l3_list2)
    i=i+1;
# =============================================================================
#     rowx=rowx+2;
#     worksheet.write_row(rowx,colx,cnam)
#     worksheet1.write_row(rowx,colx,cnam)
#     worksheet2.write_row(rowx,colx,cnam)
#     worksheet3.write_row(rowx,colx,cnam)
# =============================================================================
Prog_elapsed = round(timeit.default_timer() - Prog_start_time,3)
Prog_ed_time=datetime.now().strftime('%d-%m-%Y %H:%M:%S');
print('program st at ', Prog_st_time, 'end at ', Prog_ed_time, 'tot taken time is',Prog_elapsed)
workbook.close()
workbook1.close()
workbook2.close()
workbook3.close()
    


    