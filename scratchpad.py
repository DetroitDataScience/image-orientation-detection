#!/usr/bin/env python

import function_list as fc
from function_list import DataInfo, DataVec
import numpy as np
from prettytable import PrettyTable
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle
import operator

##### Do not change these as they are used for all the training image data #####
LoadTRNIDX = 1 # if(0): generate training image index data (sorts and assigns classes)
LoadTSTIDX = 1 # if(0): generate test image index data (sorts and assigns classes)
################################################################################
LoadTRNDATA = 1 # if(0): generate training features from images, else load features from file
LoadTSTDATA = 1 # if(0): generate test features from images, else load features from file
ShowGraphs = 0 # if(1): show LDA/PCA graphs
TrainSVM = 1 # if(0): only test the pre-trained SVMs against test data, else train SVM as well
TestSVM = 1 # if(1): run the trained SVM on the test data

n_jobs = 12 # number of parallel process to run during training parameter tuning (via cross-validation)

comparevec = np.array([[1,0,0,1,2,200,8,8,0], # use_hog, use_lum, use_pca, use_lda, num_comp, num_vec, m_block, n_block, normalize_lum
                       [1,0,0,1,3,200,8,8,0],
                       [1,0,1,1,2,200,8,8,0],
                       [1,0,1,1,3,200,8,8,0], ##
                       [0,1,0,1,2,200,8,8,0],
                       [0,1,0,1,3,200,8,8,0],
                       [0,1,1,1,2,200,8,8,0],
                       [0,1,1,1,3,200,8,8,0], ##
                       [1,1,0,1,2,200,8,8,0],
                       [1,1,0,1,3,200,8,8,0],
                       [1,1,1,1,2,200,8,8,0],
                       [1,1,1,1,3,200,8,8,0], ##
                       [1,1,0,1,2,200,8,8,1],
                       [1,1,0,1,3,200,8,8,1],
                       [1,1,1,1,2,200,8,8,1],
                       [1,1,1,1,3,200,8,8,1], ##
                       [0,1,0,1,2,200,8,8,1],
                       [0,1,0,1,3,200,8,8,1],
                       [0,1,1,1,2,200,8,8,1],
                       [0,1,1,1,3,200,8,8,1]])

classes = np.array([0, 90, 180, 270])


acc_vec = []
confusion_vec = []
report_vec = []
param_vec = []
svm_vec = []

for i in range(0,comparevec.shape[0]):
    # Feature params
    use_hog = comparevec[i,0]
    use_lum = comparevec[i,1]
    use_pca = comparevec[i,2]
    use_lda = comparevec[i,3]
    num_comp = comparevec[i,4]
    num_vec = comparevec[i,5]
    m_block = comparevec[i,6]
    n_block = comparevec[i,7]
    normalize_lum = comparevec[i,8]
    imgresize = np.array([200,200])
    hogcellsize = np.array([20,20])
    hogcellblock = np.array([1,1])
    hognumorient = 4
    
    
    P = fc.Params(classes, use_hog, use_lum, use_pca, use_lda, num_comp, num_vec, n_jobs, 
               m_block, n_block, hogcellsize, hogcellblock, hognumorient, imgresize,
               normalize_lum)
    param_vec.append(P)
    
    #################
    ##### Setup #####
    #################
    
    if TrainSVM == 1 or ShowGraphs == 1 or LoadTRNDATA == 0 or LoadTRNIDX == 0:
        TrnInfo, TrnData = fc.setupImgs(P, "train/*.JPG", "trainingdata", LoadTRNDATA, LoadTRNIDX, 0, "")
    if TestSVM == 1 or LoadTSTDATA == 0 or LoadTSTIDX == 0 or ShowGraphs == 1:
        TstInfo, TstData = fc.setupImgs(P, "test/*.JPG", "testingdata", LoadTSTDATA, LoadTSTIDX, 1, "trainingdata")
    
    
    ###################
    ##### Examine #####
    ###################  
    
    if ShowGraphs == 1:
        fc.PCA_LDA_examine(P, TrnData.feat.T, TrnData.labels, TrnData.feat.T, TrnData.labels, 'Only Trn')
        fc.PCA_LDA_examine(P, TrnData.feat.T, TrnData.labels, TstData.feat.T, TstData.labels, 'Trn/Tst')
    
    
    
    ####################
    ##### Training #####
    ####################   
    the_str = "classifiers/" + fc.ret_clstr(P, "classifier")
    if TrainSVM == 1:
        print "Training SVM"
        # prepare dimensionality reduction
        if P.use_pca == 1:
            PCA_fit = PCA(n_components=P.num_vec).fit(TrnData.feat.T)
        else:
            PCA_fit = []
        if P.use_lda == 1:
            if P.use_pca == 1:
                LDA_fit = LinearDiscriminantAnalysis(n_components=P.num_comp,store_covariance=True).fit(PCA_fit.transform(TrnData.feat.T),TrnData.labels)
                X_trn = LDA_fit.transform(PCA_fit.transform(TrnData.feat.T))
            else:
                LDA_fit = LinearDiscriminantAnalysis(n_components=P.num_comp,store_covariance=True).fit(TrnData.feat.T,TrnData.labels)
                X_trn = LDA_fit.transform(TrnData.feat.T)
        else:
            LDA_fit = []
            if P.use_pca == 1:
                X_trn = PCA_fit.transform(TrnData.feat.T)
            else:
                X_trn = TrnData.feat.T 
        
        # train SVM
        clf = OneVsRestClassifier(SVC(kernel='linear'))
        params = dict(estimator__C = [50,10,1,0.5,0.1,0.09,0.05,0.01,0.001,0.0001],
                  estimator__kernel = ["poly","linear","rbf"],
                  estimator__gamma = [0.0001,0.001,0.005,0.01,0.05,0.1,0.5,0.99],
                  estimator__degree = [2,3,4])
        tuning = GridSearchCV(clf,param_grid=params,n_jobs=P.n_jobs)
        tuning.fit(X_trn,TrnData.labels)
        
        print tuning.best_params_
        
        # save
        print "Saving %s" % the_str
        f = open(the_str, 'w')
        pickle.dump([PCA_fit,LDA_fit,tuning],f)
        f.close()
        
    
    
    ###################
    ##### Testing #####
    ###################
    
    if TestSVM == 1:
        print "Testing SVM"
        if TrainSVM == 0:
            print "Loading %s" % the_str
            f = open(the_str, 'r')
            PCA_fit, LDA_fit, tuning = pickle.load(f)
            f.close()
            
        # apply dimensionality reduction
        if P.use_pca == 1:
            X_1 = PCA_fit.transform(TstData.feat.T)
        else:
            X_1 = TstData.feat.T
        if P.use_lda == 1:
            X_tst = LDA_fit.transform(X_1)   
        else:
            X_tst = X_1
        
        svm_vec.append(tuning.best_params_)
        
        # apply SVM
        Y_pred = tuning.best_estimator_.predict(X_tst)
        
        # results
        report_vec.append(classification_report(TstData.labels,Y_pred,target_names=str(P.classes)))
        confusion_vec.append(confusion_matrix(TstData.labels,Y_pred,labels=range(len(P.classes))))
        acc_vec.append(accuracy_score(TstData.labels,Y_pred))
        

new_acc_vec = [i*100 for i in acc_vec]
t = PrettyTable(['Dim Reduction', '# of LDA Components', 'HOG', 'Lum', 'Lum+HOG', 'NormLum', 'NormLum+HOG'])
t.add_row(['LDA',2,new_acc_vec[0],new_acc_vec[4],new_acc_vec[8],new_acc_vec[16],new_acc_vec[12]])
t.add_row(['',3,new_acc_vec[1],new_acc_vec[5],new_acc_vec[9],new_acc_vec[17],new_acc_vec[13]])
t.add_row(['PCA+LDA',2,new_acc_vec[2],new_acc_vec[6],new_acc_vec[10],new_acc_vec[18],new_acc_vec[14]])
t.add_row(['',3,new_acc_vec[3],new_acc_vec[7],new_acc_vec[11],new_acc_vec[19],new_acc_vec[15]])
t.float_format = '.2'
print(t)

idx = max(enumerate(new_acc_vec), key=operator.itemgetter(1))
print "\nBest Results: %.2f" % idx[1]
print "\nSVM Params: %s" % svm_vec[idx[0]]
print "\nConfusion Matrix:"
print(confusion_vec[idx[0]])
print "\nClassification Report:"
print(report_vec[idx[0]])
