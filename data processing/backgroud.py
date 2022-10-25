import cv2
import numpy as np

h=512
w=512
img=np.zeros((h,w,3),np.uint8)+255
i=0
j=0
while i<=512:
    cv2.line(img, (i, 0), (i,512), (0, 0, 255), 3)  # 绿色，3个像素宽度
    i = i + 32

while j <=512:
    cv2.line(img, (0, j), (512, j), (0, 0, 255), 3)  # 绿色，3个像素宽度
    j = j + 32

#j = j + 32 back.jpg
#j = j + 16 back1.jpg
#j = j + 24 back2.jpg
#j = j + 64 back3.jpg

print(img.shape)
cv2.imshow('img',img)
cv2.imwrite('read.jpg',img)
cv2.waitKey(0)