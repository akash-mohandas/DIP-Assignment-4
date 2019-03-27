# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 23:52:07 2019

@author: aakas
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

zeros=[]
des_zeros=[]
kp_zeros=[]
sift1 = cv2.xfeatures2d.SIFT_create()
    
for i in range(1,6):
    raw_image = open("zero_"+str(i)+".raw",'rb').read()
    raw_image = np.frombuffer(raw_image, np.uint8)
    img = raw_image[0:28*28]
    img = np.reshape(img, (28,28))
    img = cv2.resize(img,(56,56))
    kp,des = sift1.detectAndCompute(img,None)
    kp_zeros.append(kp)      
    des_zeros.append(des)
    zeros.append(img)

ones=[]
des_ones=[]
kp_ones=[]
sift2 = cv2.xfeatures2d.SIFT_create()

for i in range(1,6):
    raw_image = open("one_"+str(i)+".raw",'rb').read()
    raw_image = np.frombuffer(raw_image, np.uint8)
    img = raw_image[0:28*28]
    img = np.reshape(img, (28,28))
    img = cv2.resize(img,(56,56))
    kp,des = sift2.detectAndCompute(img,None)
    kp_ones.append(kp)      
    des_ones.append(des)
    ones.append(img)

#kp,des = np.concatenate((kp_zeros+kp_ones[:4]),axis=0), np.concatenate((des_zeros+des_ones[:4]),axis=0)

kmeans1 = KMeans(n_clusters=2).fit(np.concatenate((des_zeros+des_ones[:4]),axis=0))
labels = kmeans1.labels_
#kmeans2 = KMeans(n_clusters=2).fit(np.array(np.concatenate(des_ones[:4])))
#labels2 = kmeans1.predict(np.array(np.concatenate(des_ones[:4])))
labels1=labels[:42]
labels2 = labels[42:]

raw_image = open("eight.raw",'rb').read()
raw_image = np.frombuffer(raw_image, np.uint8)
img = raw_image[0:28*28]
img = np.reshape(img, (28,28))
img = cv2.resize(img,(56,56))

sift3 = cv2.xfeatures2d.SIFT_create()
kp1,des1 = sift3.detectAndCompute(img,None)
#kmeans3 = KMeans(n_clusters=2).fit(des1)
labels3 = kmeans1.predict(des1)

plt.hist(labels1)
plt.show()

plt.hist(labels2)
plt.show()
plt.hist(labels3)

zeros_zeros = labels1[labels1==0].shape[0]
zeros_ones = labels1[labels1==1].shape[0]

ones_zeros = labels2[labels2==0].shape[0]
ones_ones = labels2[labels2==1].shape[0]

eight_zeros = labels3[labels3==0].shape[0]
eight_ones = labels3[labels3==1].shape[0]

zero = min(zeros_zeros,eight_zeros)/max(zeros_zeros,eight_zeros) + min(zeros_ones,eight_ones)/max(zeros_ones,eight_ones)
one = min(ones_zeros,eight_zeros)/max(ones_zeros,eight_zeros) + min(ones_ones,eight_ones)/max(ones_ones,eight_ones)

print(zero)
print(one)
if one > zero:
    print("Eight image is lassified as one")
else:
    print("Eight image is lassified as zero")
plt.show()