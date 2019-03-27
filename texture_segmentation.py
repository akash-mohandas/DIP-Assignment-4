# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 14:47:46 2019

@author: aakas
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 16:49:18 2019

@author: aakas
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 20:17:56 2019

@author: aakas
"""

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

def convolution(i,j,kernel,image):
    pix = 0.0;
    for k in range(i,i+5):
        for l in range(j,j+5):
            pix+= (image[k][l] * kernel[k-i][l-j])
            #cout<<pix<<endl
    #return (pix/sum);
    return pix;





filters = [np.array([1,4,6,4,1]).reshape(1,5),np.array([-1,-2,0,2,1]).reshape(1,5),np.array([-1,0,2,0,-1]).reshape(1,5),np.array([-1,2,0,-2,1]).reshape(1,5),np.array([1,-4,6,-4,1]).reshape(1,5)]

kernels=[]
vectors=[]
for i in range(5):
    for j in range(5):
        kernels.append(np.matmul(np.transpose(filters[i]),filters[j]))

raw_image = open("comb.raw",'rb').read()
raw_image = np.frombuffer(raw_image, np.uint8)
img = raw_image[0:510*510]
img = np.reshape(img, (510,510)) 
img = img - np.mean(img)

img_extended = np.pad(img,2,'reflect')

        #new_img = np.zeros((128,128))
features=[]
filtered_images=[]
for index,kernel in enumerate(kernels):
    filtered_img = np.zeros((510,510))
    for i in range(510):
        for j in range(510):
            filtered_img[i][j] = convolution(i,j,kernel,img_extended)
    
    filtered_images.append(filtered_img)
    
filtered_images_extended=[]
for image in filtered_images:
    Reimage = np.pad(image,6,'reflect')
    filtered_images_extended.append(Reimage)

for a in range(510):
    for b in range(510):
        features=[]
        for Reimage in filtered_images_extended:
            #features.append(np.sum(abs(Reimage[a:a+13,b:b+13])-np.mean(Reimage[a:a+13,b:b+13]))/169)
            features.append(np.sum(abs(Reimage[a:a+13,b:b+13]))/169)
    #features1=[features[0],features[6],features[12],features[18],features[24],features[1]/features[5],features[2]/features[10],features[3]/features[15],features[4]/features[20],features[19]/features[23],features[7]/features[11],features[8]/features[16],features[9]/features[21],features[13]/features[17],features[22]]
        vectors.append(np.array(features))

    #vectors.append(np.array(features))

vectors = np.array(vectors)
vectors = vectors[:,1:]


vectors = (vectors - np.mean(vectors,axis=0))/np.std(vectors,axis=0)
#vectors.dtype = 'float32'
#l,d,r = linalg.svd(vectors)
#vectors_ = np.concatenate((d[0] * l[:,0].reshape(-1,1),d[1]*l[:,1].reshape(-1,1),d[2]*l[:,2].reshape(-1,1)),axis=1)

pca = PCA(n_components = 3)
vectors_ = pca.fit_transform(vectors)

#fig = plt.figure()
#ax = fig.add_subplot(111,projection='3d')
#ax.scatter(vectors_[:,0],vectors_[:,1],vectors_[:,2])
#plt.show

kmeans = KMeans(n_clusters = 7).fit(vectors_)
final = kmeans.labels_.reshape(510,510)
for i in range(510):
    for j in range(510):
        if final[i][j]==1:
            final[i][j] = 42
        elif final[i][j]==2:
            final[i][j] = 84
        elif final[i][j]==3:
            final[i][j] = 126
        elif final[i][j]==4:
            final[i][j] = 168
        elif final[i][j]==5:
            final[i][j] = 210
        elif final[i][j]==6:
            final[i][j] = 255
final = np.uint8(final)

plt.matshow(final,cmap='gray')
plt.show()

close=np.zeros((510,510),np.uint8)
from scipy import stats
fin_ext = np.pad(final,7,'reflect')
for i in range(510):
    for j in range(510):
        close[i][j] = stats.mode(fin_ext[i:i+15,j:j+15],axis=None)[0][0]

k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
close = cv2.morphologyEx(close,cv2.MORPH_CLOSE,k,iterations=10)
plt.matshow(close,cmap='gray')

plt.show()

#f, axarr = plt.subplots(5,5)
#k=0
#for i in range(5):
#    for j in range(5):
#        axarr[i,j].imshow(filtered_images[k],cmap='gray')
     #        k+=1
#axarr[0,1].imshow(filtered_images[1],cmap='gray')
#axarr[1,0].imshow(image_datas[2])
#axarr[1,1].imshow(image_datas[3])