# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 21:52:54 2019

@author: aakas
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

raw_image1 = open("river1.raw",'rb').read()
raw_image1 = np.frombuffer(raw_image1, np.uint8)
img1 = raw_image1[0:1024*768*3]
img1 = np.reshape(img1, (1024,768,3))

img_gray1 = np.zeros((1024,768))

raw_image2 = open("river2.raw",'rb').read()
raw_image2 = np.frombuffer(raw_image2, np.uint8)
img2 = raw_image2[0:1024*768*3]
img2 = np.reshape(img2, (1024,768,3))

img_gray2 = np.zeros((1024,768))


sift = cv2.xfeatures2d.SIFT_create()
kp1,des1 = sift.detectAndCompute(img1,None)
#sift2 = cv2.xfeatures2d.SIFT_create()
kp2,des2 = sift.detectAndCompute(img2,None)

#l2_norm = [np.linalg.norm(des1[i,:]) for i in range(des1.shape[0])]
#l2_norm = np.linalg.norm(des1) for i in range(des1.shape[0])]
max_l2 = np.argmax(np.linalg.norm(des1,axis=1))
#max_l2 = l2_norm.index(max(l2_norm))

bf=cv2.BFMatcher()

matches = bf.knnMatch(des1[max_l2,:].reshape(-1,128),des2,k=2)
good=[]
for m,n in matches:
    good.append([m])

#good = sorted(good,key = lambda x: x.distance)
img3 = cv2.drawMatchesKnn(img1,[kp1[max_l2]],img2,kp2,good,None,flags=2)
(x1,y1) = kp1[good[0][0].queryIdx].pt
(x2,y2) = kp2[good[0][0].trainIdx].pt


#cv2.line(img3,(int(x1),int(y1)),(768+int(x2),int(y2)),(0,0,0),10)
img4= cv2.cvtColor(img3,cv2.COLOR_RGB2BGR)
cv2.imwrite("matches.jpg",img4)

img5 = cv2.drawKeypoints(img1,kp1,img1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img6 = cv2.drawKeypoints(img2,kp2,img2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img5= cv2.cvtColor(img5,cv2.COLOR_RGB2BGR)
img6= cv2.cvtColor(img6,cv2.COLOR_RGB2BGR)
cv2.imwrite("kp1.jpg",img5)
cv2.imwrite("kp2.jpg",img6)

plt.imshow(img5)
plt.imshow(img6)
plt.imshow(img3)
plt.show()
