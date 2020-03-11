import numpy as np
import cv2
import os
import csv
import numpy as np
from scipy import ndimage
from numpy.lib.stride_tricks import as_strided
from sklearn.linear_model import LinearRegression
import math


# rows, cols = 500, 500
# win_rows, win_cols = 5, 5

# img = np.random.rand(rows, cols)
# win_mean = ndimage.uniform_filter(img, (win_rows, win_cols))
# win_sqr_mean = ndimage.uniform_filter(img**2, (win_rows, win_cols))
# win_var = win_sqr_mean - win_mean**2


def sliding_window(arr, window_size):
    """ Construct a sliding window view of the array"""
    arr = np.asarray(arr)
    window_size = int(window_size)
    if arr.ndim != 2:
        raise ValueError("need 2-D input")
    if not (window_size > 0):
        raise ValueError("need a positive window size")
    shape = (arr.shape[0] - window_size + 1,
             arr.shape[1] - window_size + 1,
             window_size, window_size)
    if shape[0] <= 0:
        shape = (1, shape[1], arr.shape[0], shape[3])
    if shape[1] <= 0:
        shape = (shape[0], 1, shape[2], arr.shape[1])
    strides = (arr.shape[1]*arr.itemsize, arr.itemsize,
               arr.shape[1]*arr.itemsize, arr.itemsize)
    return as_strided(arr, shape=shape, strides=strides)

def cell_neighbors(arr, i, j, d):
    """Return d-th neighbors of cell (i, j)"""
    w = sliding_window(arr, 2*d+1)

    ix = np.clip(i - d, 0, w.shape[0]-1)
    jx = np.clip(j - d, 0, w.shape[1]-1)

    i0 = max(0, i - d - ix)
    j0 = max(0, j - d - jx)
    i1 = w.shape[2] - max(0, d - i + ix)
    j1 = w.shape[3] - max(0, d - j + jx)

    return w[ix, jx][i0:i1,j0:j1].ravel()


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
    filename1 = "/home/students/jhuang/Videocoding/Crew_1280x720_60Hz.yuv"
    filename2 = "/home/students/jhuang/Videocoding/Crewwoflash_1280x720_60Hz.yuv"
    size = (720, 1280)
    cap1 = VideoCaptureYUV(filename1, size)
    cap2 = VideoCaptureYUV(filename2, size)
    count = 0
    for i in range (600):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if ret1:
            cv2.imshow('frame1', frame1)
            cv2.imshow('frame2', frame2)
            if i in index:
                print(i)
                flash = frame1
                noflash = frame2 
                
                ScaleY = np.zeros((720,1280))
                OffsetY = np.zeros((720,1280))
                Y1 = np.zeros((720,1280))
                Y2 = np.zeros((720,1280))
                # ScaleU = np.zeros((720,1280))
                # OffsetU = np.zeros((720,1280))

                for m in range(0,720):
                    for n in range(0,1280):
                        Y1[m,n] = flash[m,n,0]
                        Y2[m,n] = noflash[m,n,0]
                for r in range(0,flash.shape[0]):
                    for c in range(0,flash.shape[1]):
                        I_flash = np.reshape(cell_neighbors(Y1, r, c, d=8),(-1,1))
                        I_noflash = np.reshape(cell_neighbors(Y2, r, c, d=8),(-1,1))
                        regY = LinearRegression().fit(I_flash, I_noflash)
                        ScaleY[r,c] = regY.coef_[0]
                        OffsetY[r,c] = regY.intercept_[0]
                noflashcomp = noflash
                print(ScaleY.shape)
                print(OffsetY.shape)
                Y2 = Y1 * ScaleY + OffsetY
                noflashcomp[...,0] = Y2
                # noflashcomp = cv2.cvtColor(noflashcomp, cv2.COLOR_YUV2BGR)
                cv2.imwrite(os.path.join('/home/students/jhuang/crewwoflash2','%.6dcomp.png'%i), noflashcomp)

                cv2.imshow('noflashcomp',noflashcomp)
                cv2.waitKey(1)

        else:
            break
      




