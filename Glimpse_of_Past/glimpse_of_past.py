import cv2
import numpy as np
import os
from matplotlib import pyplot as plt


##1. Load both images
##past = cv2.imread('./past.jpg',1)
##present = cv2.imread('./present.jpeg',1)

##present = cv2.imread('./purdue1.jpg',1)
##past = cv2.imread('./purdue2.PNG',1)

past = cv2.imread('./past.jpg',1)
present = cv2.imread('./16.Purdue-Memorial-Union.jpg',1)

g1 = cv2.cvtColor(past, cv2.COLOR_BGR2GRAY)
g2 = cv2.cvtColor(present, cv2.COLOR_BGR2GRAY)


# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(g1,None)
kp2, des2 = sift.detectAndCompute(g2,None)

pt1 = np.float32([kp.pt for kp in kp1])
pt2 = np.float32([kp.pt for kp in kp2])

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
goodmatch = []

# Applying Lowe ratio test
for m in matches:
    if len(m) == 2 and m[0].distance:
        goodmatch.append((m[0].trainIdx , m[0].queryIdx))

if len(goodmatch) > 4:
    pt1 = np.float32([pt1[i] for (_,i) in goodmatch])
    pt2 = np.float32([pt2[i] for (i,_) in goodmatch])
    (H, mask) = cv2.findHomography(pt1,pt2, cv2.RANSAC, 4.0)
# Apply ratio test
##goodmatch = []
##for m,n in matches:
##    if (m.distance < 0.75*n.distance):
##        goodmatch.append((m[0].train.Idx , m[0].query.Idx))
##        
##if len(goodmatch) > 4: #required to find homography 
##    pt1 = np.float32([pt1[i] for (_,i) in goodmatch])
##    pt2 = np.float32([pt2[i] for (i,_) in goodmatch])
##    (H, mask) = cv2.findHomography(pt1,pt2, cv2.RANSAC, 4.0)
##    

 
result = cv2.warpPerspective(past, H , (800, 800) )


for i in range(0,present.shape[0]):
    for j in range(0,present.shape[1]):
        if(np.all(result[i][j]) == 0):
            result[i][j] = present[i][j]

for i in range(0,result.shape[0]):
    for j in range(0, result.shape[1]):
        if(i>present.shape[0] or j>present.shape[1]):
                  result[i][j] = [0,0,0]
           
            
            
cv2.imshow('result',result)            
# cv2.drawMatchesKnn expects list of lists as matches.
##img3=cv2.drawMatchesKnn(g1,kp1,g2,kp2,goodmatch,None,flags=2)
##plt.imshow(img3),plt.show()
cv2.imshow('present',present)
cv2.waitKey(0)
cv2.destroyAllWindows()
