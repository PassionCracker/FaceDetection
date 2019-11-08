import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("zuker.jpg",1) #1=> color img, 0=>b/w img
#open cv loads image in BGR format. To see exact, convert into RBG format
plt.imshow(img)  #type of img is nd-numpy array

img.shape
print(img)
# Get clear idea on numpy

exact_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(exact_img)
haar_cascade_face =cv2.CascadeClassifier("haar.xml")
faces_rects = haar_cascade_face.detectMultiScale(exact_img, scaleFactor = 1.2, minNeighbors = 5)
plt.imshow(exact_img, cmap='gray')

print(faces_rects)
print(len(faces_rects))

for (x,y,w,h) in faces_rects:
     cv2.rectangle(exact_img, (x, y), (x+w, y+h), (150, 250, 50), 5)
plt.imshow(exact_img)