#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cam = cv2.VideoCapture(0)

while True:

    ret, im = cam.read()

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.2,5)

    for(x,y,w,h) in faces:

        cv2.rectangle(im, (x-20,y-20), (x+w+20, y+h+20), (0,255,0), 4)

    cv2.imshow('Anass Face Detection', im)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

