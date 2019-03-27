# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 13:06:00 2019

@author: aakas
"""

import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from scipy import linalg

def convolution(i,j,kernel,image):
    pix = 0.0;
    for k in range(i,i+5):
        for l in range(j,j+5):
            pix+= (image[k][l] * kernel[k-i][l-j])
            #cout<<pix<<endl
    #return (pix/sum);
    return pix;




raw_images=[]
for filename in glob.iglob("hw4_images/**"):
    raw_image = open(filename,'rb').read()
    raw_image = np.frombuffer(raw_image, np.uint8)
    img = raw_image[0:128*128]
    img = np.reshape(img, (128,128))
    raw_images.append(img)
    #plt.matshow(img,cmap='gray')

filters = [np.array([1,4,6,4,1]).reshape(1,5),np.array([-1,-2,0,2,1]).reshape(1,5),np.array([-1,0,2,0,-1]).reshape(1,5),np.array([-1,2,0,-2,1]).reshape(1,5),np.array([1,-4,6,-4,1]).reshape(1,5)]

kernels=[]
vectors=[]
for i in range(5):
    for j in range(5):
        kernels.append(np.matmul(np.transpose(filters[i]),filters[j]))
        
for img in raw_images:
    #img = img - np.mean(img)
    new_img = np.pad(img,7,'reflect')
    avg_img=np.zeros((128,128))
    for a in range(128):
        for b in range(128):
            avg_img[a][b] = img[a][b] - np.mean(new_img[a:a+15,b:b+15])
    img_extended = np.pad(avg_img,2,'reflect')
    #new_img = np.zeros((128,128))
#    for j in range(2):
#        for i in range(2,2+128):
#            img_extended[j][i] = img[2-j][i-2]
#    
#    for j in range(2):
#        for i in range(2,2+128):
#            img_extended[128+j+2][i] = img[128-j-2][i-2]
#        
#    for i in range(2):
#        for j in range(2,2+128):
#            img_extended[j][i] = img[j-2][2-i]
#        
#    for i in range(2):
#        for j in range(2,2+128):
#            img_extended[j][128+2+i] = img[j-2][128-i-2]
#            
#    img_extended[0][0] = img_extended[0][4]
#    img_extended[0][1] = img_extended[0][3]
#    img_extended[1][0] = img_extended[1][4]
#    img_extended[1][1] = img_extended[1][3]
#    img_extended[0][128+2] = img_extended[0][128]
#    img_extended[0][128+3] = img_extended[0][128-1]
#    img_extended[1][128+2] = img_extended[1][128]
#    img_extended[1][128+3] = img_extended[1][128-1]
#    img_extended[128+2][0] = img_extended[128+2][4]
#    img_extended[128+2][1] = img_extended[128+2][3]
#    img_extended[128+3][0] = img_extended[128+3][4]
#    img_extended[128+3][1] = img_extended[128+3][3]
#    img_extended[128+2][128+2] = img_extended[128+2][128]
#    img_extended[128+2][128+3] = img_extended[128+2][128-1]
#    img_extended[128+3][128+2] = img_extended[128+3][128]
#    img_extended[128+3][128+3] = img_extended[128+3][128-1]
#	
#    for j in range(128):
#        for i in range(128):
#            img_extended[j+2][i+2] = img[j][i]
    features=[]
    for index,kernel in enumerate(kernels):
        energy = 0
        for i in range(128):
            for j in range(128):
                res = convolution(i,j,kernel,img_extended)
                
                #if index==0:
                 #   res/=256
                #energy+=(res*res)
                energy+=abs(res)
        energy/=(128*128)
        features.append(energy)
    
    #features1=[features[0],features[6],features[12],features[18],features[24],(features[1]+features[5])/2,(features[2]+features[10])/2,(features[3]+features[15])/2,(features[4]+features[20])/2,(features[19]+features[23])/2,(features[7]+features[11])/2,(features[8]+features[16])/2,(features[9]+features[21])/2,(features[13]+features[17])/2,(features[14]+features[22])/2]
    #vectors.append((np.array(features)-np.mean(features))/np.std(features))
    vectors.append(features)
    #vectors.append(np.array(features))

#vectors = np.array(vectors)
#print("Max values")
#print(np.argmax(vectors[:,1:],axis=1))
#print("Min values")
#print(np.argmin(vectors[:,1:],axis=1))
print("Variance: ", np.var(vectors,axis=0))

import pandas as pd
df=pd.DataFrame(vectors,columns=['l5l5','l5e5','l5s5','l5w5','l5r5','e5l5','e5e5','e5s5','e5w5','e5r5','s5l5','s5e5','s5s5','s5w5','s5r5','w5l5','w5e5','w5s5','w5w5','w5r5','r5l5','r5e5','r5s5','r5w5','r5r5'])
df.to_excel('vectors.xlsx')
    

vectors = (vectors - np.mean(vectors,axis=1).reshape(-1,1))/np.std(vectors,axis=1).reshape(-1,1)
#vectors = (vectors - np.mean(vectors,axis=0))/np.std(vectors,axis=0)

#pca = PCA(n_components = 3)
#vectors_ = pca.fit_transform(vectors)
l,d,r = linalg.svd(vectors)
vectors_ = np.concatenate((d[0] * l[:,0].reshape(-1,1),d[1]*l[:,1].reshape(-1,1),d[2]*l[:,2].reshape(-1,1)),axis=1)

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(vectors_[:,0],vectors_[:,1],vectors_[:,2])
plt.show

kmeans = KMeans(n_clusters = 4).fit(vectors_)

print('0')
for k in range(12):
    if kmeans.labels_[k]==0:
        plt.matshow(raw_images[k],cmap='gray')
        
print('1')
for k in range(12):
    if kmeans.labels_[k]==1:
        plt.matshow(raw_images[k],cmap='gray')
        
print('2')
for k in range(12):
    if kmeans.labels_[k]==2:
        plt.matshow(raw_images[k],cmap='gray')
        
print('3')
for k in range(12):
    if kmeans.labels_[k]==3:
        plt.matshow(raw_images[k],cmap='gray')

plt.show()

        