import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
import csv
import glob
import math
from PIL import Image
import sys
import pandas as pd

class VideoCaptureYUV:
    def __init__(self, filename, size):
        self.height, self.width = size
        self.frame_len = self.width * self.height * 3 // 2
        self.f = open(filename, 'rb')
        self.shape = (int(self.height*1.5), self.width)

    def read_raw(self):
        try:
            raw = self.f.read(self.frame_len)
            yuv = np.frombuffer(raw, dtype=np.uint8)
            yuv = yuv.reshape(self.shape)
        except Exception as e:
            print(str(e))
            return False, None
        return True, yuv

    def read(self):
        ret, yuv = self.read_raw()
        if not ret:
            return ret, yuv
        bgr = cv.cvtColor(yuv, cv.COLOR_YUV2BGR_I420, 3)
        return ret, bgr

with open('./test2.csv','r') as csvfile:
  reader = csv.reader(csvfile)
  column = [row[1] for row in reader]
  column.pop(0)
  column = list(map(int,column))

with open('./test3.csv','r') as csvfile1:
  reader1 = csv.reader(csvfile1)
  column1 = [row[1] for row in reader1]
  column1.pop(0)
  column1 = list(map(int,column1))
index = [x for x, e in enumerate(column1) if e != 0]
print(index)
# deltay = np.zeros((720,1280))
# deltau = np.zeros((720,1280))
# deltav = np.zeros((720,1280))
# ye = np.zeros((720,1280))
# ue = np.zeros((720,1280))
# ve = np.zeros((720,1280))



if __name__ == "__main__":
    filename = "Crew_1280x720_60Hz.yuv"
    size = (720, 1280)
    cap = VideoCaptureYUV(filename, size)
    ret, frame = cap.read()
    prev_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    previous_img = frame
    count = 0
    k = 0
    count = 0
    fps=25
    fourcc= cv.VideoWriter_fourcc('m','p','4','v')
    videowriter=cv.VideoWriter('./test.mp4',fourcc,fps,(1280,720))

    flows = []

    # x = np.arange(0,1280)
    # y = np.arange(0,720)

    # while(cap.isOpened()):
    for i in range(50):
        print(i)
        previous_img = frame.copy()
        # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
        ret, frame = cap.read()
        img = frame
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)
        # gray = yuv[...,0]
        U = yuv[...,1]  #0.492 * (img[...,0] - gray)
        V = yuv[...,2]  #0.877 * (img[...,2] - gray)

        prev_yuv = cv.cvtColor(previous_img, cv.COLOR_BGR2YUV)
        # prev_gray = prev_yuv[...,0]
        prev_U = prev_yuv[...,1]
        prev_V = prev_yuv[...,2]
        

        # Calculates dense optical flow by Farneback method
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0) 
        flows.append(flow)
        # RV=np.arange(720,3,-4)
        # CV=np.arange(3,1280,4)
        # u,v = np.meshgrid(CV, RV)
        # print(k)
        # fig, ax = plt.subplots()
        # x = flow[..., 0][::4, ::4]
        # y = flow[..., 1][::4, ::4]
        # q = ax.quiver(u,v,x, y,color='red',headlength=5)
        # plt.show()
        


        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

        

        # Frames are read by intervals of 1 millisecond. The programs breaks out of the while loop when the user presses the 'q' key
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        img = frame
        if column[i+1] == 1:
            imgcomp = img
            for m in range (720):
                for n in range (1280):
                    flow[m, n, 1] = - flow[m, n, 1] + m
                    flow[m, n, 0] = - flow[m, n, 0] + n
            imgcomp = cv.remap(previous_img, flow[..., 0], flow[..., 1], interpolation = cv.INTER_CUBIC)

            videowriter.write(imgcomp)
        
        elif column1[i] == 0 & column1[i+1] == 1:
            m = n-1
            imgcomp = img
            for m in range (720):
                for n in range (1280):
                    flow[m, n, 1] = - flow[m, n, 1] + m
                    flow[m, n, 0] = - flow[m, n, 0] + n
            imgcomp = cv.remap(previous_img, flow[..., 0], flow[..., 1], interpolation = cv.INTER_CUBIC)

            videowriter.write(imgcomp)
        elif column1[i-1] == 0 & column1[i] == 1 & column1[i+1] == 1:
            prev_flow = flows[i-1]
            imgcomp = cv.remap(previous_img, prev_flow[..., 0], prev_flow[..., 1], interpolation = cv.INTER_CUBIC)
            for m in range (720):
                for n in range (1280):
                    flow[m, n, 1] = - flow[m, n, 1] + m
                    flow[m, n, 0] = - flow[m, n, 0] + n
            imgcomp = cv.remap(imgcomp, flow[..., 0], flow[..., 1], interpolation = cv.INTER_CUBIC)
            videowriter.write(imgcomp)
        else:
            videowriter.write(img)

        prev_gray = gray
        # cv.imwrite('%.1dillumotioncomp.png'%count,imgcomp)
        count += 1
        k += 1
        

    # The following frees up resources and closes all windows
    cap.release()
    cv.destroyAllWindows()