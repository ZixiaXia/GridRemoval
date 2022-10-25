import numpy as np
import cv2 as cv
import random
h=700
w=700
h2=512
w2=512
t=1
w1=int(w/3)
h1=int(h/2)
x=h-h2
y=w-w2
# 随机数组
list1 = [0, 1, 2]
list2 = [0, 1]
def min(x,y):
    if int(x/2)>int(y/3):
        return y/3
    else:
        return x/2
print('Hello World!')
while t<=140:
    # 创建一个全白图像
    img = np.zeros((h, w, 1), np.int)+255
    #创建随机数
    r1 = np.random.randint(1, w1, 12, int)
    r2 = np.random.randint(1, h1, 12, int)
    r3 = np.random.randint(int(min(h, w) / 4), int(min(h, w)/1.5), 3, int)  # 半径半轴
    r4 = np.random.randint(2, 5, 5, int) # 线宽145 2-5
    # 圆心2
    rc1 = random.randint(int(w / 4), int(w / 2))
    rc2 = random.randint(int(h / 4), int(h / 2))
    rc3 = random.randint(int(w / 2), w-50)
    rc4 = random.randint(int(h / 2), h-50)
    # 随机数组，随机排序
    random.shuffle(list1)
    random.shuffle(list2)
    # 绘制一条直线段
    cv.line(img, (r1[0], r2[0]), (r1[1]+w1*list1[0], r2[1]+h1*list2[0]), (0, 0, 0), r4[0], cv.LINE_8)#起、止、色、宽（-1为填充）。
    #绘制一个矩形
    cv.rectangle(img,(r1[2]+2*w1,r2[2]+h1),(r1[3]+w1*list1[1],r2[3]+h1*list2[1]),(0,0,0),r4[1],cv.LINE_8)#起左上、止右下、色、宽（-1为填充）
    #绘制一个圆
    cv.circle(img,(rc1,rc2),r3[0],(0,0,0),r4[2],cv.LINE_8)#圆心、半径、色、宽（-1为填充）
    #绘制椭圆
    cv.ellipse(img,(rc3,rc4),(r3[1],r3[2]),0,0,360,(0,0,0),r4[3])#圆心、长短轴长度、
    #绘制三角形
    cv.line(img, (r1[6]+2*w1, r2[6]), (r1[7]+w1*list1[2], r2[7]+h1*list2[0]), (0, 0, 0), r4[4], cv.LINE_8)  # 起、止、色、宽（-1为填充）。
    cv.line(img, (r1[7]+w1*list1[2], r2[7]+h1*list2[0]), (r1[8]+w1*list1[2], r2[8]+h1*list2[0]), (0, 0, 0), r4[4], cv.LINE_8)  # 起、止、色、宽（-1为填充）。
    cv.line(img, (r1[8]+w1*list1[2], r2[8]+h1*list2[0]), (r1[6]+2*w1, r2[6]), (0, 0, 0), r4[4], cv.LINE_8)  # 起、止、色、宽（-1为填充）。
    #绘制菱形
    cv.line(img, (r1[9], r2[9] + h1), (r1[10], r2[10] + h1), (0, 0, 0), r4[4],cv.LINE_8)  # 起、止、色、宽（-1为填充）。
    cv.line(img, (r1[10], r2[10] + h1), (r1[11] + w1 * list1[2], r2[11] + h1 * list2[1]), (0, 0, 0), r4[4],cv.LINE_8)  # 起、止、色、宽（-1为填充）。
    cv.line(img, (r1[9], r2[9] + h1), ((r1[9]+r1[11]-r1[10]) + w1 * list1[2], (r2[9]+r2[11]-r2[10]) + h1 * list2[1]), (0, 0, 0), r4[4],cv.LINE_8)  # 起、止、色、宽（-1为填充）。
    cv.line(img, ((r1[9]+r1[11]-r1[10]) + w1 * list1[2], (r2[9]+r2[11]-r2[10]) + h1 * list2[1]), (r1[11] + w1 * list1[2], r2[11] + h1 * list2[1]), (0, 0, 0), r4[4],cv.LINE_8)  # 起、止、色、宽（-1为填充）。
    #剪切原图
    c1 = np.random.randint(0, x, 5)
    c2 = np.random.randint(0, y, 5)
    for i in range(5):
        cropimg=img[c1[i]:c1[i]+h2,c2[i]:c2[i]+w2,:]
        #cv.imwrite('dataset/output/crop' + str(t) + '-' + str(i) + '.jpg', cropimg)
        cv.imwrite('datatest/output/crop' + str(5*(t - 1) + i) + '.jpg', cropimg)
        # #归一化
        # result = np.zeros(cropimg.shape, dtype=np.float32)
        # cv.normalize(cropimg, result, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        # cv.imwrite('dataset/output-n/crop' + str(t) + '-' + str(i) + '.jpg', result)
    # 增加白噪声
    for i in range(1300):
        pos1 = np.random.randint(1, h - 1, dtype=np.int)
        pos2 = np.random.randint(1, w - 1, dtype=np.int)
        pos3 = np.random.randint(1, 15, dtype=np.int)
        img[pos1:pos1 + pos3, pos2:pos2 + pos3, :] = 255
    # 剪切噪声图
    for i in range(5):
        cropimg=img[c1[i]:c1[i]+h2,c2[i]:c2[i]+w2,:]
        #cv.imwrite('dataset/input/crop'  + str(t) + '-' + str(i) + '.jpg', cropimg)
        cv.imwrite('datatest/input/crop' + str(5*(t - 1) + i) + '.jpg', cropimg)
    t=t+1