import cv2
import numpy as np

image=cv2.imread('myImage.png')
cv2.imshow('image',image)
cv2.waitKey(0)
image2=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# image[(image[:,:,0]<200) | (image[:,:,1]<200) | (image[:,:,2]<200)]=0
kernal=np.ones((8,8))*1/64;
imageblurr=cv2.filter2D(image2,-1,kernal);
cv2.imshow('image',imageblurr)
cv2.waitKey(0)

# capture=cv2.VideoCapture('dog.mp4');
#
# while True:
#     isTrue, frame=capture.read();
#     cv2.imshow('image',frame);
#
#     if cv2.waitKey(20) & 0xFF==ord('d'):
#         break;
#
# capture.release()
# cv2.destroyAllWindows()