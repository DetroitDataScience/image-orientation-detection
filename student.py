#!/usr/bin/env python

import function_list as fc
from function_list import DataInfo, DataVec
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle

##### Do not change these as they are used for all the training image data #####
LoadTRNIDX = 1 # if(0): generate training image index data (sorts and assigns classes)
LoadTSTIDX = 1 # if(0): generate test image index data (sorts and assigns classes)
################################################################################
LoadTRNDATA = 1 # if(0): generate training features from images, else load features from file
LoadTSTDATA = 1 # if(0): generate test features from images, else load features from file
ShowGraphs = 1 # if(1): show LDA/PCA graphs
TrainSVM = 1 # if(0): only test the pre-trained SVMs against test data, else train SVM as well
TestSVM = 1 # if(1): run the trained SVM on the test data

n_jobs = 1 # number of parallel process to run during training parameter tuning (via cross-validation)

classes = np.array([0, 90, 180, 270]) # rotation classes 0 deg -> 270 deg

# Feature params
m_block = 8 # Luminance cell MxN extraction, number of blocks vertically
n_block = 8 # Luminance cell MxN extraction, number of blocks horizontally
imgresize = np.array([200,200]) # resize the image before extracting features
hogcellsize = np.array([20,20]) # HOG cell size (in pixels)
hogcellblock = np.array([1,1]) # HOG cell block normalization 
hognumorient = 4 # HOG number of gradient orientation bins for the histogram of each cell

######################## 
## Need to Set values ##
########################
use_hog = 0 # if(1): use HOG features
use_lum = 0 # if(1): use Luminance Moments as features
use_pca = 0 # if(1): use PCA as a dimensionality reduction technique
use_lda = 0 # if(1): use LDA as a dimensionality reduction technique
num_comp = 0 # the number of LDA components to use
num_vec = 0 # the number of PCA vectors to use
normalize_lum = 0 # Normalize the luminance features
######################## 
######################## 


# concatenate all parameters
P = fc.Params(classes, use_hog, use_lum, use_pca, use_lda, num_comp, num_vec, n_jobs, 
           m_block, n_block, hogcellsize, hogcellblock, hognumorient, imgresize,
           normalize_lum)


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
    clf = OneVsRestClassifier(SVC())
    
    ######################## 
    ## Need to Set values ##
    ########################
    params = dict(estimator__C = [],
              estimator__kernel = [],
              estimator__gamma = [],
              estimator__degree = [])
    ######################## 
    ######################## 
    
    tuning = GridSearchCV(clf,param_grid=params,n_jobs=P.n_jobs)
    tuning.fit(X_trn,TrnData.labels)
    
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
    
    # apply SVM
    Y_pred = tuning.best_estimator_.predict(X_tst)
    
    # results
    print "\nResults: %.2f" % 100 * accuracy_score(TstData.labels,Y_pred)
    print "\nSVM Params: %s" % tuning.best_params_
    print "\nConfusion Matrix:"
    print(confusion_matrix(TstData.labels,Y_pred,labels=range(len(P.classes))))
    print "\nClassification Report:"
    print(classification_report(TstData.labels,Y_pred,target_names=str(P.classes)))
