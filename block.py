import numpy as np
import cv2
import os
import csv


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
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420, 3)
        return ret, bgr
with open('./test.csv','r') as csvfile:
  reader = csv.reader(csvfile)
  column = [row[1] for row in reader]
  column.pop(0)
  column = list(map(int,column))

index = [x for x, e in enumerate(column) if e != 0]

deltay = np.zeros((720,1280))
deltau = np.zeros((720,1280))
deltav = np.zeros((720,1280))
ye = np.zeros((720,1280))
ue = np.zeros((720,1280))
ve = np.zeros((720,1280))

if __name__ == "__main__":
  filename1 = "Crew_1280x720_60Hz.yuv"
  filename2 = "Crewwoflash_1280x720_60Hz.yuv"
  size = (720, 1280)
  cap1 = VideoCaptureYUV(filename1, size)
  cap2 = VideoCaptureYUV(filename2, size)
  count = 0
  for n in range (600):
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if ret1:
      if n in index:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        yuv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2YUV)
        U1 = yuv1[...,1]  #0.492 * (img[...,0] - gray)
        V1 = yuv1[...,2]  #0.877 * (img[...,2] - gray)
        yuv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2YUV)
        U2 = yuv2[...,1]  #0.492 * (img[...,0] - gray)
        V2 = yuv2[...,2]  #0.877 * (img[...,2] - gray)
        cv2.imshow('frame1',frame1)
        cv2.imshow('frame2',frame2)
        cv2.waitKey(0)
        # cv2.imwrite(os.path.join('/home/students/jhuang/crewwflash', '%.6d.png'%i),frame)
        for i in range (720):
          for j in range (1280):
            if gray1[i][j] > gray2[i][j]:
              deltay[i][j] = gray1[i][j] - gray2[i][j]
              ye[i][j] = 1
            else:
              deltay[i][j] = gray2[i][j] - gray1[i][j]
              ye[i][j] = -1
            if U1[i][j] > U2[i][j]:
              deltau[i][j] = U1[i][j] - U2[i][j]
              ue[i][j] = 1
            else:
              deltau[i][j] = U2[i][j] - U1[i][j]
              ue[i][j] = -1
            if V1[i][j] > V2[i][j]:
              deltav[i][j] = V1[i][j] - V2[i][j]
              ve[i][j] = 1
            else:
              deltav[i][j] = V2[i][j] - V1[i][j]
              ve[i][j] = -1
            # print(prev_img)
        for y in range (720):
          for x in range (1280):
                    # pixelindexy[y][x] = ye[y][x]*deltay[y][x]
                    # pixelindexu[y][x] = ue[y][x]*deltau[y][x]
                    # pixelindexv[y][x] = ve[y][x]*deltav[y][x]-
            if deltay[y][x] >= 0 and ye[y][x] > 0: #and ve[y][x] <= 0 and ue[y][x] >= 0:  #:
                        # print(y,x)          
              frame1[y,x,2] = frame1[y,x,2] - ye[y][x]*deltay[y][x] - ve[y][x]*1.14*deltav[y][x]
              frame1[y,x,1] = frame1[y,x,1] - ye[y][x]*deltay[y][x] + ue[y][x]*0.395*deltau[y][x] + ve[y][x]*0.581*deltav[y][x]
              frame1[y,x,0] = frame1[y,x,0] - ye[y][x]*deltay[y][x] - ue[y][x]*2.033*deltau[y][x]
        cv2.imwrite('%.1dillumotioncomp.png'%n,frame1)
        # width = size[0]
        # height = size[1]
        # # Define the window size
        # windowsize_r = 64
        # windowsize_c = 64
        # for r in range(0,gray1.shape[0]):
        #   for c in range(0,gray1.shape[1]):
        #     window1 = gray1[r:r+windowsize_r,c:c+windowsize_c]
        #     window2 = gray2[r:r+windowsize_r,c:c+windowsize_c]
        #     print(r,c)
        #     # print('-------',window1)
        #     print('+++++++',window2)
        #     w = np.mean(window1)
        #     # i = r/windowsize_r
        #     # j = c/windowsize_c
        #     # blocky[i][j] = blocky[i][j] + w
    else:
      break
      
  # blocky = numpy.zeros((height/windowsize_r, width/windowsize_c))
# import glob
# from PIL import Image
# for frames in glob.glob('./1/*.jpg'):
#   img = cv2.imread(frames)
#   gray_levels = 256
#   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
#   im = Image.open(frames)
#   pix = im.load()
#   width = im.size[0]
#   height = im.size[1]
  
#   # Define the window size
#   windowsize_r = 16
#   windowsize_c = 16

#   blocky = numpy.zeros((height/windowsize_r, width/windowsize_c))

#   # The average luminance component (Y) of an entire frame
#   y1 = numpy.mean(gray)

#   framey.append(y1)

  

#   # Each frame is partitioned into blocks
#   # for r in range(0,gray.shape[0] - windowsize_r, windowsize_r):
#   #   for c in range(0,gray.shape[1] - windowsize_c, windowsize_c):
  # for r in range(0,gray.shape[0], windowsize_r):
  #   for c in range(0,gray.shape[1], windowsize_c):
  #     window = gray[r:r+windowsize_r,c:c+windowsize_c]
  #     w = numpy.mean(window)
  #     i = r/windowsize_r
  #     j = c/windowsize_c
  #     blocky[i][j] = blocky[i][j] + w
        
#       # The blocks are sorted in decreasing order 
#       w1 = numpy.sort(blocky)
  
#   print(frameindex)  
#   frameindex += 1   
  
#   print(blocky)