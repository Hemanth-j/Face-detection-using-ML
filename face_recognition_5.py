"""#predicting model
import joblib
model=joblib.load("model.pkl")
print(model)
import face_recognition_4
xx=face_recognition_4.x
yy=face_recognition_4.y
#print(xx)
import numpy as np
vbc=xx.iloc[58,:]
vbc=np.array(vbc).reshape(1,-1)
print(face_recognition_4.dframe)
print(yy.iloc[58,:])
print(model.predict(vbc))"""
import numpy as np
import pywt
import cv2
import joblib
model=joblib.load("model.pkl")
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
t_image_path=str(input("enter the image path"))
x=[]
img = cv2.imread(t_image_path)
scalled_raw_img = cv2.resize(img, (32, 32))
img_har = w2d(t_image_path, 'db1',5)
scalled_img_har = cv2.resize(img_har, (32, 32))
combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))
x.append(combined_img)
x = np.array(x).reshape(len(x),4096).astype(float)
c=model.predict(x)
print(c)
#{"messi ": 0, "nirmala sitharaman ": 1, "ratan tata ": 2, "virat kohli  ": 3}
if(c==0):
    name=str("messi ")
elif(c==1):
    name = str("nirmala sitharaman ")
elif(c==2):
    name = str("ratan tata ")
else:
    name=str("virat kohli  ")
img = cv2.imread(t_image_path, cv2.IMREAD_ANYCOLOR)
print(name)
while True:
   xyz_img=cv2.putText(img,name,(80,50),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0),4)
   img2=cv2.resize(xyz_img,(600,600))
   cv2.imshow("face recognition",img2)
   cv2.waitKey(30)
cv2.destroyAllWindows()


