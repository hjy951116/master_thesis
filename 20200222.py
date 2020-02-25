import numpy as np
import cv2
import os
import csv
import numpy as np
from scipy import ndimage
from numpy.lib.stride_tricks import as_strided
from sklearn.linear_model import LinearRegression
import math
import glob



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


with open('./test.csv','r') as csvfile:
  reader = csv.reader(csvfile)
  column = [row[1] for row in reader]
  column.pop(0)
  column = list(map(int,column))

index = [x for x, e in enumerate(column) if e != 0]

if __name__ == "__main__":
    path1 = '/home/students/jhuang/crew_A/'
    fileList1 = os.listdir(path1)
    fileList1.sort()
    path2 = '/home/students/jhuang/crew_B/'
    fileList2 = os.listdir(path2) 
    fileList2.sort()
    for i in range (len(index)-1):
        img1 = path1 + os.sep + fileList1[i]
        frame1 = cv2.imread(img1)
        # cv2.imshow('frame1', frame1)
        print(img1)
        img2 = path2 + os.sep + fileList2[i]
        frame2 = cv2.imread(img2)
        print(img2)
        flash = cv2.cvtColor(frame1, cv2.COLOR_BGR2YUV)
        noflash = cv2.cvtColor(frame2, cv2.COLOR_BGR2YUV)
        # cv2.imshow('noflash', cv2.cvtColor(noflash, cv2.COLOR_YUV2BGR))
        # cv2.imshow('flash', cv2.cvtColor(flash, cv2.COLOR_YUV2BGR))
            
        cv2.waitKey(1)
        ScaleY1 = np.zeros((720,1280))
        ScaleY2 = np.zeros((720,1280))
        OffsetY1 = np.zeros((720,1280))
        OffsetY2 = np.zeros((720,1280))
        for r in range(0,flash.shape[0]):
            for c in range(0,flash.shape[1]):
                I_flash = np.reshape(cell_neighbors(flash[...,0], r, c, d=1),(-1,1))
                I_noflash = np.reshape(cell_neighbors(noflash[...,0], r, c, d=1),(-1,1))
                regY1 = LinearRegression().fit(I_flash, I_noflash)
                regY2 = LinearRegression().fit(I_noflash, I_flash)
                ScaleY1[r,c] = regY1.coef_[0]
                ScaleY2[r,c] = regY2.coef_[0]
                OffsetY1[r,c] = regY1.intercept_[0]
                OffsetY2[r,c] = regY2.intercept_[0]
        noflash1 = cv2.cvtColor(noflash, cv2.COLOR_YUV2BGR)
        flash1 = cv2.cvtColor(flash, cv2.COLOR_YUV2BGR)
        noflashcomp1 = noflash
        noflashcomp2 = noflash
        flashcomp = flash
        print(ScaleY1.shape)
        print(OffsetY1.shape)
        noflashcomp1[...,0] = flash[...,0] * ScaleY1 + OffsetY1
        # flashcomp[...,0] = noflash[...,0] * ScaleY2 + OffsetY2
        # flashcomp[...,0] = (noflash[...,0] - OffsetY1) / ScaleY1
        noflashcomp1 = cv2.cvtColor(noflashcomp1, cv2.COLOR_YUV2BGR)
        # flashcomp = cv2.cvtColor(flashcomp, cv2.COLOR_YUV2BGR)
        cv2.imshow('noflashcomp1',noflashcomp1)    
        diff1 = cv2.absdiff(noflash1, noflashcomp1)
        # diff1 = noflash1 - noflashcomp
        # diff2 = cv2.absdiff(flash1, flashcomp)
        # for r in range(0,flash.shape[0]):
        #     for c in range(0,flash.shape[1]):
        #         if diff2[r,c][...,0] > 35:
        #             noflashcomp[r,c] = noflashcomp[r,c] + diff1[r,c]

        # diff1 = noflash1 - noflashcomp
        # diff2 = flash1 - flashcomp
        # noflashcomp = noflashcomp - diff2
        noflashcomp2[...,0] = (flash[...,0] - OffsetY2) / ScaleY2
        noflashcomp2 = cv2.cvtColor(noflashcomp2, cv2.COLOR_YUV2BGR)

        cv2.imshow('noflashcomp2',noflashcomp2)
        diff2 = cv2.absdiff(noflash1, noflashcomp2)
        diff = cv2.absdiff(noflashcomp1, noflashcomp2)
        diff3 = cv2.absdiff(diff2, diff)
        cv2.imshow('diff1',diff1)
        cv2.imshow('diff2',diff2)
        cv2.imshow('diff',diff)
        cv2.imshow('diff3',diff3)

        # cv2.imshow('flashcomp',flashcomp)
        # cv2.imshow('diff1',diff1)
        # cv2.imshow('diff2',diff2)
        cv2.waitKey(0)