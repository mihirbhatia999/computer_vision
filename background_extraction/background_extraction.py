import numpy as np
import cv2
import glob

#capture video 
cap = cv2.VideoCapture('C:/Users/Mihir/Desktop/Purdue IE/Fall 2017/Academics/IE590/Homework/homework 2/q2/4p-c1 (1).avi')

#use mixture of gaussian to extract background 
fgbg = cv2.createBackgroundSubtractorMOG2()

i=1                                                                         #initialising variable for filey
while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    cv2.imshow('frame',fgmask)

    print('background %d read'%i)
    cv2.imwrite(r'.\Color\color %d'%i + '.JPG', frame)
    cv2.imwrite(r'.\BGD\backgd %d'%i +'.JPG',fgmask)
    i=i+1
    print('images read %d'%i)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release
cv2.destroyAllWindows()

#condition where the pixel always remains black. Just extract all those pixel values

bgd = []
for  bgd in glob.glob('C:/Users/Mihir/Desktop/Purdue IE/Fall 2017/Academics/IE590/Homework/homework 2/BGD/*'):
    img = cv2.imread(bgd,1)
    bgd.append(img)

color = []
for  color in glob.glob('C:/Users/Mihir/Desktop/Purdue IE/Fall 2017/Academics/IE590/Homework/homework 2/Color/*'):
    img2 = cv2.imread(color,1)
    color.append(img2) 
                
height, width, channels = frame.shape

background = np.zeros((height,width))

for px in range(0,height) :
    
    for py range(0,width) :         #pixel chosen with coordinates (px,py) 

        for j in range(100,i):              #image chosen

            for k in range( j - 3 , j +3 ):     #analysising few images before and after to check if the pixel is actually background

                sum  =  sum + bgd[ j ][px,py]
                
            if( sum  ==  0 ):
                background[ px , py ] = color [ j ][px , py]

            sum=0

cv2.imshow('background' , background)


                

            

      
            

        










