# Image Orientation Detection
A presentation and exercise designed as an introduction into machine learning.  Presented at the [Detroit Data Science Meetup on 4/20/2017](https://www.meetup.com/Detroit-Data-Science-Meetup/events/238413214/).

Checkout how to [get started](https://github.com/DetroitDataScience/image-orientation-detection/wiki/Getting-Started) in the wiki.


# Exercise Overview

The goal of this exercise is to demonstrate some of the topics presented in the talk by training and testing an image orientation classifier using different features and dimensionality reduction techniques.

 * 4 possible classes: 0, 90, 180, and 270 degree rotation
 * The image features are already provided from the [training and test datasets](https://github.com/DetroitDataScience/image-orientation-detection-data) 
   * Histogram of Oriented Gradients (HOG)
   * Spatial Color Moments (3 mean and 3 variance values of L, U, and V color spaces)
   * Normalized Spatial Color Moments
 * Dimensionality reduction techniques already provided are:
   * Principal Component Analysis (PCA)
   * Linear Discriminant Analysis (LDA)
 
The task is to choose the combination of features and parameter values for training a Support Vector Machine (SVM) for classification. The only code that _needs_ to be edited is in student.py is designated with the following:


```python
######################## 
## Need to Set values ##
########################

... code here ...

######################## 
######################## 
```

## Training and Testing data

The training dataset contains 2149 images of people on bikes taken during a marathon. There is variation in pose, lighting, scale, the number of people in each image, and the background (e.g., mountains, foilage, road, etc.). The training set image data is 708x1064 pixels and are all already correctly oriented at 0 degrees. 

The test dataset contains 626 images of people running/walking taken during a marathon (i.e., in the same style as the training set). There is variation in pose, lighting, scale, image dimensions, the number of people in each image, and the background. The test set image data is 708x1064 pixels or 1064x708 pixels and is similarly already correctly oriented at 0 degrees. 

To create the separate classes, the training and test data was randomly split into quarters where each set is rotated accordingly before feature extraction. The files img_idx/trainingdata.pckl and img_idx/testingdata.pckl contain the image index information relating which image is assigned to which class before rotation and feature extraction.


## Feature extraction 

Features have already been extracted using the settings within student.py and saved to files within the folder features.

 * Training data is designated as `trainingdata_*`
 * Testing data is designated as `testingdata_*`

To designate which set of features are saved:

 * `*_lum_*` designates the use of Spatial Color Moments (and *_nolum_* means these features are not included)
 * `*_normlum_*` designates the use of Normalized Spatial Color Moments
 * `*_hog_*` designates the use of HOG features (and *_nohog_* means these features are not included)
 * `*_#x#.pckl` designates the number of rows and columns which were used to divide the image into patches to produce the Spatial Color Moment features

The the files `*_nolum_nohog_*` do not exist because that would mean there are no features extracted from the images. To choose which set of features to load, simply set the variables in student.py:

```python
use_hog = 0 # if(1): use HOG features
use_lum = 0 # if(1): use spatial color moment as features
normalize_lum = 0 # if(1): normalize the spatial color moment features (only works if use_lum=1)
```

## Dimensionality Reduction

While not limited to only using PCA and/or LDA, these two dimensionality reduction techniques were included in student.py and their use be toggled accordingly:

```python
use_pca = 0 # if(1): use PCA as a dimensionality reduction technique
use_lda = 0 # if(1): use LDA as a dimensionality reduction technique
num_comp = 0 # the number of LDA components to use
num_vec = 0 # the number of PCA vectors to use
```

Some functions are included to help visualize the effect of using PCA and/or LDA on the features to examine.


## SVM training

After loading features and choosing the dimensionality reduction technique(s) and their parameterizations, the SVM needs to be trained to separate the image orientation classes. The student.py code includes functionality to perform grid search cross-validation of varying SVM parameters. However, the possible values need to be filled in:

```python
params = dict(estimator__C = [],
          estimator__kernel = [],
          estimator__gamma = [],
          estimator__degree = [])
```


# Additional Information

A small collection of conference and journal papers found using Google Scholar (related to the task of image orientation detection) can be found in the papers folder. Also an overview of SVMs and the feature extraction approaches can be found in ppt/features.pdf. 

The goal of the exercise is to provide easy to use start-up code, data, and a baseline result to make it easy for you to work on this outside of the meetup. 
