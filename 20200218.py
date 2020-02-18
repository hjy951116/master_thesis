import numpy as np
import cv2
import os
import csv
import numpy as np
from scipy import ndimage 


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
  filename1 = "Crew_1280x720_60Hz.yuv"
  filename2 = "Crewwoflash_1280x720_60Hz.yuv"
  size = (720, 1280)
  cap1 = VideoCaptureYUV(filename1, size)
  cap2 = VideoCaptureYUV(filename2, size)
  count = 0
  for n in range (100):
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
        cv2.waitKey(1)
        win_rows, win_cols = 3, 4
        win_mean1[...,0] = ndimage.uniform_filter(frame1[...,0], (win_rows, win_cols))
        win_sqr_mean1[...,0] = ndimage.uniform_filter(frame1[...,0]**2, (win_rows, win_cols))
        win_var1[...,0] = win_sqr_mean1 - win_mean1**2
        win_mean2[...,0] = ndimage.uniform_filter(frame2[...,0], (win_rows, win_cols))
        win_sqr_mean2[...,0] = ndimage.uniform_filter(frame2[...,0]**2, (win_rows, win_cols))
        win_var2[...,0] = win_sqr_mean2 - win_mean2**2
        S[...,0] = ((frame1[...,0]-win_mean1[...,0])*(frame2[...,0]-win_mean2[...,0]))/np.square((frame1[...,0]-win_mean1[...,0]))
        C[...,0] = win_mean1[...,0] - win_mean2[...,0]
        print(C[...,0])
        print(S[...,0])
        Ref[...,0] = (frame1[...,0]-C[...,0])/S[...,0]
        cv2.imshow('Ref',Ref)
        cv2.waitKey(0)
        

        # for r in range(0,gray1.shape[0]):
        #   for c in range(0,gray1.shape[1]):
        #     window1 = cell_neighbors(gray1, r, c, d=1)
        #     window2 = cell_neighbors(gray2, r, c, d=1)
        #     print(r,c)
        #     print('-------',window1)
        #     print('+++++++',window2)
        #     w = np.mean(window1)
            # i = r/windowsize_r
            # j = c/windowsize_c
            # blocky[i][j] = blocky[i][j] + w
    else:
      break
      




