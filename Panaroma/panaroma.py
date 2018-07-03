import cv2
import numpy as np
import os


####1. Load both images, and convert to grayscale.
im_left_o = cv2.imread('./uttower_left.JPG',1)
im_right_o = cv2.imread('./uttower_right.JPG',1)

im_left = im_right_o
im_right = im_left_o

im_left_g = cv2.cvtColor(im_left, cv2.COLOR_BGR2GRAY)
im_right_g = cv2.cvtColor(im_right, cv2.COLOR_BGR2GRAY)
print('step 1 done')


##2. Detect feature points in both images. You can use a corner detector, SIFT or SURF for
##that purpose.
sift = cv2.xfeatures2d.SIFT_create()
kp1 , des1 = sift.detectAndCompute(im_left_g,None)
kp2 , des2 = sift.detectAndCompute(im_right_g,None)


##3. Extract local neighborhoods around every keypoint in both images, and form
##descriptors simply by "flattening" the pixel values in each neighborhood to onedimensional
##vectors. Experiment with different neighborhood sizes to see which one
##works the best.
pt1 = np.float32([kp.pt for kp in kp1])
pt2 = np.float32([kp.pt for kp in kp2])
print('step 2, step 3 done')


##4. Compute distances between every descriptor in one image and every descriptor in the
##other image (Euclidean distance). Alternatively, experiment with computing
##normalized correlation, or Euclidean distance after normalizing all descriptors to have
##zero mean and unit standard deviation.
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
goodmatch = []

# Applying Lowe ratio test
for m in matches:
    if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
        goodmatch.append((m[0].trainIdx , m[0].queryIdx))
        
print('step 4 done')

            
##5. Select putative matches based on the matrix of pairwise descriptor distances obtained
##above. You can select all pairs whose descriptor distances are below a specified
##threshold, or select the top few hundred descriptor pairs with the smallest pairwise
##distance
if len(goodmatch) > 4:
     
####6. Run RANSAC to estimate a homography mapping one image onto the other. Report
####the number of inliers and the average residual for the inliers (squared distance
####between the point coordinates in one image and the transformed coordinates of the
####matching point in the other image). Also, display the locations of inlier matches in
####both images.

    pt1 = np.float32([pt1[i] for (_,i) in goodmatch])
    pt2 = np.float32([pt2[i] for (i,_) in goodmatch])
    (H, mask) = cv2.findHomography(pt1,pt2, cv2.RANSAC, 4.0)

#to find the inliers and outliers 
t=0
for k in mask:
    if k == 0:
        t = t+1
        
print('\n')
print(H)
print("homography matrix found")
print('\n')
print('the number of outliers = %d'%t)
print('the number of inliers = %d'%(k-t))

#height and width of the 2 images 
left_wd = im_left_g.shape[1]
left_ht = im_left_g.shape[0]

right_wd =im_right_g.shape[1]
right_ht = im_right_g.shape[0]

##(Bonus)_Warp one image onto the other using the estimated transformation. You can
##refer to Matlab’s maketform and imtransform functions, but your work should be in
##Python.

##8. (Bonus) Create a new image big enough to hold the panorama and composite the two
##images into it. You can composite by simply averaging the pixel values where the two
##images overlap. Your result should look something like this (but hopefully with a more
##precise alignment):

result = cv2.warpPerspective(im_left , H , (left_wd + right_wd , left_ht))

result[0:right_ht , 0: right_wd] = im_right
cv2.imshow('result2', result)
cv2.imshow('left', im_left)
cv2.imshow('right', im_right)
cv2.waitKey(0)
