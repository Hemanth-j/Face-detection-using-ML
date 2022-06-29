#generating cropped faces
import cv2
from PIL import Image
import matplotlib.pyplot as plt
def img_to_cropped(xyz):
    img=cv2.imread(xyz)
    grayy=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#plt.imshow(gray,cmap="gray")
#plt.show()
    face=cv2.CascadeClassifier("C:\\Users\\HEMANTH\\Downloads\\haarcascade_frontalface_default.xml")
#eye= cv2.CascadeClassifier('C:\\Users\\HEMANTH\\Downloads\\haarcascade_eye.xml')
    face_detect=face.detectMultiScale(grayy)
#eye_detect=eye.detectMultiScale(grayy,2.5,5)
    x,y,w,h=face_detect[0]
#a,b,c,d=eye_detect[0]
    bh=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    return img[y:y+h,x:x+w]
#for (x, y, w, h) in eye_detect:
#    cv2.rectangle(img, (x, y), (x + w, y + h), (56, 0, 255), 3)
"""while True :
  bh=cv2.resize(bh,(700,700))
  cv2.imshow("cropped",bh)
  faces=img[y:y+h,x:x+w]
  cv2.imshow("xcropped",faces)
  if cv2.waitKey(1) == ord('q'):
      break
print(face_detect)"""
#print(eye_detect)
import os
from_path="C:\\Users\\HEMANTH\\Downloads\\face recongnition\\dataset\\"
to_path="C:\\Users\\HEMANTH\\Downloads\\face recongnition\\cropped1\\"
os_dir=[]
for i in os.scandir(from_path):
  if i is not dir():
    os_dir.append(i.path)
#print(os_dir)
if os.path.exists(to_path):
  os.rmdir(to_path)
else:
  os.mkdir(to_path)
cropped_dir_names=[]
celebs_name={}
for img_path in os_dir:
  count=1
  celeb=img_path.split("\\")[-1]
  celeb=celeb.split("-")[0]
  print(celeb)
  for j in os.scandir(img_path):
    roi_color=img_to_cropped(j.path)
    #print(j)
    if roi_color is not None :
       cropped_folder=to_path + celeb + "\\"
       if not os.path.exists(cropped_folder):
           os.mkdir(cropped_folder)
           cropped_dir_names.append(cropped_folder)
           print("generating cropped images :",cropped_folder)
       cropped_file_name=str(celeb) + str(count) +str(".png")
       cropped_file_path=str(cropped_folder)+str("\\")+str(cropped_file_name)
       cv2.imwrite(cropped_file_path,roi_color)
       #celebs_name[celeb].append(cropped_file_path)
       count=count+1
