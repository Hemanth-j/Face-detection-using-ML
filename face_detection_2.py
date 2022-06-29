#generating wavelet transform image and data preprocessing
import os
import matplotlib.pyplot as plt
dict={}
for i in os.scandir("C:\\Users\\HEMANTH\\Downloads\\face recongnition\\cropped1"):
     arr = []
     #print(i.name)
     for j in os.scandir(i):
        arr.append(j.path)
     dict[i.name]=arr
#print(dict.get("messi "))
#print(dict.items())

import numpy as np
import pywt
import cv2
def w2d(img, mode='haar', level=1):
    imArray =cv2.imread(img)
    imArray = cv2.cvtColor(imArray,cv2.COLOR_RGB2GRAY )
    imArray =  np.float32(imArray)
    imArray /= 255;
    coeffs=pywt.wavedec2(imArray, mode, level=level)
    coeffs_H=list(coeffs)
    coeffs_H[0]*= 0;
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)
    return imArray_H
"""pathh="C:\\Users\\HEMANTH\\Downloads\\face recongnition\\cropped1\\ratan tata \\{}2.png"
im_har = w2d(pathh,'db1',5)
plt.subplot(2,2,1)
plt.imshow(im_har,cmap="gray")
plt.subplot(2,2,2)
x=plt.imread(pathh)
plt.imshow(x,cmap="gray")
plt.show()"""
x,y=[],[]
for celebs_name,img_path in dict.items():
    for t_image_path in img_path:
        t_image_path = t_image_path.replace("\\", "\\\\")
        img = cv2.imread(t_image_path)
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(t_image_path, 'db1',5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))
        x.append(combined_img)
        y.append(celebs_name)
x = np.array(x).reshape(len(x),4096).astype(float)
#print(x[0])
#print(x.shape)
#print(x)
import pandas as pd
dff=pd.DataFrame(x)
dff["target"]=y
#print(dff.shape)
dff["Target"]=dff["target"].map({"messi ":0,"nirmala sitharaman ":1,"ratan tata ":2,"virat kohli  ":3})
df=dff.drop(["target"],axis="columns")
#print(df)


