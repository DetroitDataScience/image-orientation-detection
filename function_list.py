#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
from skimage import io, color, exposure
from skimage.filters import gaussian
from skimage.feature import hog
from skimage.transform import resize
from sklearn.feature_extraction import image
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import glob
import pickle
import copy

plt.rcParams.update({'figure.max_open_warning':0})




class Params():
    def __init__(self, classes, use_hog, use_lum, use_pca, use_lda, num_comp, 
                 num_vec, n_jobs, m_block, n_block, hogcellsize, hogcellblock,
                 hognumorient, imgresize, normalize_lum):
        self.classes = classes
        self.use_hog = use_hog
        self.use_lum = use_lum
        self.use_pca = use_pca
        self.use_lda = use_lda
        self.num_comp = num_comp
        self.num_vec = num_vec
        self.n_jobs = n_jobs
        self.m_block = m_block
        self.n_block = n_block
        self.hogcellsize = hogcellsize
        self.hogcellblock = hogcellblock
        self.hognumorient = hognumorient
        self.imgresize = imgresize
        self.normalize_lum = normalize_lum


class DataInfo():
    def __init__(self, img_list, class_vec):
        self.img_list = img_list
        self.img_idx = []
        self.class_idx = []
        self.class_vec = class_vec
        
    def generate_idx(self):
        self.img_idx = np.arange(len(self.img_list))
        np.random.shuffle(self.img_idx)
        self.class_idx = np.round(np.linspace(0,len(self.img_list),len(self.class_vec)+1))


class DataVec():
    def __init__(self, img_min, img_max, feat, labels, rot):
        self.img_min = img_min
        self.img_max = img_max
        self.feat = feat
        self.labels = labels
        self.rot = rot


def ret_svstr( P, sv_str ):
    featstr1 = 'lum'
    if P.use_lum == 0:
        featstr1 = 'nolum'
    else:
        if P.normalize_lum == 1:
            featstr1 = 'normlum'
    featstr2 = 'hog'
    if P.use_hog == 0:
        featstr2 = 'nohog'
    return sv_str + "_" + featstr1 + "_" + featstr2 + "_" + str(P.m_block) + "x" + str(P.n_block) + ".pckl"


def ret_clstr( P, sv_str ):
    featstr1 = 'pca' + str(P.num_vec)
    if P.use_pca == 0:
        featstr1 = 'nopca'
    featstr2 = 'lda' + str(P.num_comp)
    if P.use_lda == 0:
        featstr2 = 'nolda'
    return sv_str + "_" + featstr1 + "_" + featstr2 + "_" + ret_svstr(P, '')


def defineImgIdx( P, dir_str, sv_str, svloadflag ):
    the_str = "img_idx/" + sv_str + ".pckl"
    if svloadflag == 0: # setup and save
        infostor = DataInfo([],P.classes)
            
        # List of images
        infostor.img_list = []
        for filename in glob.glob(dir_str):
            infostor.img_list.append(filename)
        
        infostor.generate_idx()
        print "Saving %s" % the_str
        f = open(the_str, 'w')
        pickle.dump(infostor,f)
        f.close()
    else: # load
        print "Loading %s" % the_str
        f = open(the_str, 'r')
        infostor = pickle.load(f)
        f.close()
    return infostor
    

def setupImgs( P, dir_str, sv_str, svloadflag, svloadflagidx, trntstflag, trn_sv_str = "" ):
    infostor = defineImgIdx(P, dir_str, sv_str, svloadflagidx)
    the_str = "features/" + ret_svstr(P, sv_str)
    if svloadflag == 0: # setup and save        
        datastor = DataVec([],[],[],[],[])
        
        # For each class prepare images
        img_min = []
        img_max = []
        if trntstflag == 1 and P.normalize_lum == 1:
            the_str1 = "features/" + ret_svstr(P, trn_sv_str)
            print "Loading %s" % the_str1
            f = open(the_str1, 'r')
            datastorTrn = pickle.load(f)
            f.close()
            datastorTrn.feat = None # clear some memory
            img_min = datastorTrn.img_min
            img_max = datastorTrn.img_max
        if P.use_hog == 0:
            hogsz = 0
        else:
            hogsz = (P.hognumorient*np.prod((P.imgresize/P.hogcellsize)/P.hogcellblock))
        if P.use_lum == 0:
            lumsz = 0
        else:
            lumsz = (6*P.m_block*P.n_block)
        featvecsize = lumsz + hogsz
        datastor.feat = np.zeros((featvecsize,len(infostor.img_list)))
        datastor.labels = np.zeros(len(infostor.img_list))
        tmpidx = 0
        for i in range(0,len(infostor.class_vec)):
            # load images
            startnum = infostor.class_idx[i].astype(int) + tmpidx
            stopnum = infostor.class_idx[i+1].astype(int)
            imgnum = infostor.img_idx[startnum:stopnum]

            tmp = computeFeatureVectorBatch(P, infostor.img_list, imgnum, infostor.class_vec[i], img_min, img_max)
            if trntstflag == 0:                
                if tmpidx == 0:
                    datastor.img_min = copy.deepcopy(tmp.img_min)
                    datastor.img_max = copy.deepcopy(tmp.img_max)
                else:
                    datastor.img_min += tmp.img_min
                    datastor.img_max += tmp.img_max
            datastor.feat[:,startnum:stopnum] = tmp.feat.copy()
            datastor.labels[startnum:stopnum] = tmp.labels.copy()
            datastor.rot.append(tmp.rot)
            tmpidx = 1
            
        if trntstflag == 0:
            datastor.img_min /= len(infostor.class_vec)
            datastor.img_max /= len(infostor.class_vec)
        print "Saving %s" % the_str
        f = open(the_str, 'w')
        pickle.dump(datastor,f)
        f.close()
    else: # load
        print "Loading %s" % the_str
        f = open(the_str, 'r')
        datastor = pickle.load(f)
        f.close()
        
    return infostor, datastor


def computeFeatureVectorBatch( P, img_list, img_idx, rot, img_min = [], img_max = [] ):   
    imgcol = io.imread_collection([img_list[k] for k in img_idx])
    if P.use_hog == 0:
        hogsz = 0
    else:
        hogsz = (P.hognumorient*np.prod((P.imgresize/P.hogcellsize)/P.hogcellblock))
    if P.use_lum == 0:
        lumsz = 0
    else:
        lumsz = (6*P.m_block*P.n_block)
    featvecsize = lumsz + hogsz
    datastor = DataVec([],[],[],[],rot)
    datastor.feat = np.zeros((featvecsize,len(img_idx)))
    datastor.labels = (datastor.rot/90)*np.ones(len(img_idx))
    
    for i in range(0,len(img_idx)):
        img = exposure.equalize_adapthist(imgcol[i])
        img = resize(img,P.imgresize)        
        # rotate as needed
        if datastor.rot != 0:
            img = ndimage.rotate(img,datastor.rot)   
        
        if P.use_hog == 1:
            # extract hog features
            img1 = color.rgb2gray(img)
            img1 = gaussian(img1,sigma=2)
            hogarray = hog(img1, orientations=P.hognumorient, pixels_per_cell=P.hogcellsize, cells_per_block=P.hogcellblock)
        if P.use_lum == 1:
            # extract LUV moment features
            img2 = color.rgb2luv(img)
            patches = image.extract_patches_2d(img2, (P.m_block, P.n_block))
            # compute mean and variance for each channel
            pmean = np.mean(patches,0)
            pvar = np.var(patches,0)       
            # concatenate into feature vector
            feat = np.concatenate((pmean,pvar),2)
        if P.use_hog == 1 and P.use_lum == 1:
            datastor.feat[:,i] = np.concatenate((feat.flatten(),hogarray),0)
        elif P.use_hog == 1 and P.use_lum == 0:
            datastor.feat[:,i] = hogarray
        elif P.use_hog == 0 and P.use_lum == 1:
            datastor.feat[:,i] = feat.flatten()
    # normalize
    if P.normalize_lum == 1:
        if len(img_min) == 0:
            if P.use_lum == 0:
                img_min = np.zeros(datastor.shape[0])
            else:
                img_min = np.nanmin(datastor.feat,1)
                if P.use_hog == 1:
                    img_min[lumsz:lumsz+hogsz] = 0
            datastor.img_min = img_min
        if len(img_max) == 0:
            if P.use_lum == 0:
                img_max = np.zeros(datastor.shape[0])
            else:
                img_max = np.nanmax(datastor.feat,1)
                if P.use_hog == 1:
                    img_max[lumsz:lumsz+hogsz] = 0
            datastor.img_max = img_max    
        tmp1 = (img_max - img_min)
        if P.use_hog == 1:
            tmp1[lumsz:lumsz+hogsz] = 1
        tmp = np.tile(tmp1,(len(img_idx),1)).T
        datastor.feat = (datastor.feat - np.tile(img_min,(len(img_idx),1)).T) / tmp
    return datastor


def plotscatter( P, fig, X, labels, title_str, legendflag = 1 ):
    color_vec = np.array(["r","g","b","y"])
       
    for j in range(0,len(P.classes)):
        if X.shape[1] == 3:
            fig.scatter(X[labels==j,0], X[labels==j,1], X[labels==j,2], c=color_vec[j], label=P.classes[j], edgecolors='black')
        else:
            plt.scatter(X[labels==j,0], X[labels==j,1], color=color_vec[j], label=P.classes[j], edgecolors='black')  
    
    if X.shape[1] == 3:
        fig.set_title(title_str)
        if legendflag == 1:
            fig.legend(loc=8,bbox_to_anchor=(0,-0.3,1,.1), mode="expand",ncol=len(P.classes),numpoints=1,handlelength=0)
    else:
        plt.title(title_str)
        if legendflag == 1:
            plt.legend(loc=8,bbox_to_anchor=(0,-0.3,1,.1), mode="expand",ncol=len(P.classes),numpoints=1,handlelength=0)
    return


def PCA_LDA_examine( P, feat1, labels1, feat2, labels2, title_str ):

    # Just LDA
    LDA_fit_X = LinearDiscriminantAnalysis(n_components=P.num_comp,store_covariance=True).fit(feat1,labels1)
    X = LDA_fit_X.transform(feat2)
    fig = plt.figure(figsize=(4,3))
    if P.num_comp == 3:
        ax = fig.add_subplot(111, projection='3d')
        plotscatter(P, ax, X, labels2, "Only LDA: " + title_str)
    else:
        plotscatter(P, fig, X, labels2, "Only LDA: " + title_str)
    
    # Applying PCA then LDA
    if P.num_comp == 2:
        fig = plt.figure(figsize=(19,5))
    else:
        fig = plt.figure(figsize=(19,10))
    if P.use_hog == 1 and P.use_lum == 1: 
        pca_vec = np.array([5,10,20,50,100,200,300,400,500])
    else:
        pca_vec = np.array([5,10,20,50,100,200,300])
    ct = 1
    for i in range(0,len(pca_vec)):
        PCA_fit_X = PCA(n_components=pca_vec[i]).fit(feat1)
        LDA_fit_X = LinearDiscriminantAnalysis(n_components=P.num_comp,store_covariance=True).fit(PCA_fit_X.transform(feat1),labels1)
        X = LDA_fit_X.transform(PCA_fit_X.transform(feat2))
        if P.num_comp == 2:
            fig.add_subplot(2,5,ct)
            plotscatter(P, fig, X, labels2, '# of PCA components = ' + str(pca_vec[i]), 0)
        else:
            ax = fig.add_subplot(3,4,ct, projection='3d')
            plotscatter(P, ax, X, labels2, '# of PCA components = ' + str(pca_vec[i]), 0)
        ct += 1
    return





