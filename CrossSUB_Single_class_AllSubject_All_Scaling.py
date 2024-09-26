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
labe="CorssSub_"+sff+"_"+"(19f)";
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
cnam=['Sno','TrActive','TrDrowsy','TesActive','TesDrowsy','','Classificatoin'];
for ScNam, Scmethod  in ScalingMethods:
    cnam.append(ScNam)
worksheet.write_row(rowx,colx,cnam)
worksheet1.write_row(rowx,colx,cnam)
worksheet2.write_row(rowx,colx,cnam)
worksheet3.write_row(rowx,colx,cnam)

Prog_start_time = timeit.default_timer();
Prog_st_time=datetime.now().strftime('%d-%m-%Y %H:%M:%S');

from sklearn.model_selection import KFold
kfold = KFold(len(l), True, 1)

# =============================================================================
# handling nanvalues
# =============================================================================
def handlingDataframenan(mydataset):
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
    return mydataset;
# =============================================================================
# handling inf values
def handlingInf(mydata):
    p=mydata;
    for j in range(19):
        index = 0
        for i in p[:,j]:
            if not np.isfinite(i):
                print("feture",j,'col',index,'  ', i)
                mydata[index,j]=(mydata[index-1,0]+mydata[index-2,0])/2;
            index +=1
    return mydata
# =============================================================================
for name, clf  in models:
    i=1;
    for train, test in kfold.split(data):
        traindata=data[train]
        testdata=data[test]
        #    print('Total No of Train files',len(traindata))
        #    print('Total No of Test files',len(testdata))
        myset=[];
        '''My Training data'''
        for f in range(len(traindata)):
            mydataset=pd.read_csv(traindata[f])
            #        print(f,':-',traindata[f],'Samples= ',mydataset.shape[0], end='')
            if f==0:
                df1=mydataset.copy()
                continue
            df1 = df1.append([mydataset])
            #        print(' Till The Sample ',df1.shape[0])
        from sklearn.utils import shuffle
        mydataset = shuffle(df1)
        mydataset=handlingDataframenan(mydataset)
        [rc, cc]=mydataset.shape;
            #array = dataframe.values
        MyTrainingX=[];
        MyTrainingY=[];
        MyTrainingX = mydataset.iloc[:,:-1].values
        MyTrainingX=handlingInf(MyTrainingX);
        MyTrainingY=mydataset.iloc[:,cc-1].values
        W=pd.DataFrame(MyTrainingY)
        labelencoder_X=LabelEncoder()
        MyTrainingY=labelencoder_X.fit_transform( MyTrainingY)
        from collections import Counter
        Trainvalues=Counter(MyTrainingY)
        
        '''My Testing data'''
        for f in range(len(testdata)):
            mydataset=pd.read_csv(testdata[f])
            #        print(f,':-',testdata[f],'Samples= ',mydataset.shape[0], end='')
            df1 = mydataset
            #        print(' Till The Sample ',df1.shape[0])
            #        from sklearn.utils import shuffle
            mydataset = shuffle(df1)
            mydataset=handlingDataframenan(mydataset)
            [rc, cc]=mydataset.shape;
            #array = dataframe.values
        MytestdataX=[];
        MytestdataY=[];
        MytestdataX = mydataset.iloc[:,:-1].values
        MytestdataX=handlingInf(MytestdataX )
        MytestdataY=mydataset.iloc[:,cc-1].values
        W=pd.DataFrame(MytestdataY) 
        labelencoder_X=LabelEncoder()
        MytestdataY=labelencoder_X.fit_transform( MytestdataY)
        #    from collections import Counter
        Testvalues=Counter(MytestdataY)
        list2=[i,Trainvalues[0],Trainvalues[1],Testvalues[0],Testvalues[1],'',name]
        l1_list2=[i,Trainvalues[0],Trainvalues[1],Testvalues[0],Testvalues[1],'',name]
        l2_list2=[i,Trainvalues[0],Trainvalues[1],Testvalues[0],Testvalues[1],'',name]
        l3_list2=[i,Trainvalues[0],Trainvalues[1],Testvalues[0],Testvalues[1],'',name]
        ''' Model Selectin''' 
        MyTrainingX=np.nan_to_num(MyTrainingX)
        MytestdataX=np.nan_to_num(MytestdataX)
        for ScNam, Scmethod  in ScalingMethods:
            #-----------------------------
            ''' Scaling'''
            if ScNam is 'WithoutScaling':
                MyTrainingXMain=[];
                MytestdataXMain=[];
                MyTrainingXMain=MyTrainingX;
                MytestdataXMain=MytestdataX;

            else:
                MyTrainingXMain=[];
                MytestdataXMain=[];
                MyTrainingXMain=Scmethod.fit_transform(MyTrainingX);
                MytestdataXMain=Scmethod.fit_transform(MytestdataX);

        #-----------------------------
            print (i,Trainvalues[0],Testvalues[1],'C:',name,' S:',ScNam, end='')
            if name is 'ANN':
                print('Fe ',MyTrainingXMain.size,"*",'Tar',MyTrainingY.size)
                clf.fit(MyTrainingXMain,MyTrainingY, nb_epoch=100, batch_size=5)
                y_pred_class = clf.predict_classes(MytestdataXMain)
            else:
                clf.fit( MyTrainingXMain,MyTrainingY) 
                y_pred_class=clf.predict(MytestdataXMain)
            totacu=round( (metrics.accuracy_score(MytestdataY,  y_pred_class)*100),3);
            totMisacu=round( (1-metrics.accuracy_score(MytestdataY,  y_pred_class))*100,3);
            sensitivityVal=round( (metrics.recall_score(MytestdataY,y_pred_class))*100,3); 
            precision=round((metrics.precision_score(MytestdataY,y_pred_class))*100,3);
            f1score=round(2*((sensitivityVal*precision)/(sensitivityVal+precision)),3);
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

    
    rowx=rowx+2;
    worksheet.write_row(rowx,colx,cnam)
    worksheet1.write_row(rowx,colx,cnam)
    worksheet2.write_row(rowx,colx,cnam)
    worksheet3.write_row(rowx,colx,cnam)
#    i=i+1;
Prog_elapsed = round(timeit.default_timer() - Prog_start_time,3)
Prog_ed_time=datetime.now().strftime('%d-%m-%Y %H:%M:%S');
print('program st at ', Prog_st_time, 'end at ', Prog_ed_time, 'tot taken time is',Prog_elapsed)
workbook.close()
workbook1.close()
workbook2.close()
workbook3.close()
    


    