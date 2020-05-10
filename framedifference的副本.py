import cv2
import os
import math
from skimage import measure
import numpy as np
from matplotlib import pyplot as plt
import csv
from sklearn.linear_model import LinearRegression
import imutils

# class VideoCaptureYUV:
#     def __init__(self, filename, size):
#         self.height, self.width = size
#         self.frame_len = self.width * self.height * 3 // 2
#         self.f = open(filename, 'rb')
#         self.shape = (int(self.height*1.5), self.width)
#
#     def read_raw(self):
#         try:
#             raw = self.f.read(self.frame_len)
#             yuv = np.frombuffer(raw, dtype=np.uint8)
#             yuv = yuv.reshape(self.shape)
#
#         except Exception as e:
#             print(str(e))
#             return False, None
#         return True, yuv
#
#     def read(self):
#         ret, yuv = self.read_raw()
#         if not ret:
#             return ret, yuv
#         bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420, 3)
#         return ret, bgr
#
# def psnr(img1, img2):
#     mse = numpy.mean((img1 - img2) ** 2)
#     if mse == 0:
#         return 100
#     PIXEL_MAX = 255.0
#     return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
# ssim = []
# ssima = []
# frameindex = []
# xa = []
# with open('./test.csv','r') as csvfile:
#   reader = csv.reader(csvfile)
#   column = [row[1] for row in reader]
#   column.pop(0)
#   column = list(map(int,column))
#
# index = [x for x, e in enumerate(column) if e != 0]
# if __name__ == "__main__":
#     filename = "/Users/hjyyyyy/Desktop/code2-master/Crew_1280x720_60Hz.yuv"
#     size = (720, 1280)
#     cap = VideoCaptureYUV(filename, size)
#     ret, previous_frame = cap.read()
#     count = 0
#     while 1:
#         if ret:
#             for count in range (599):
#                 print(count)
#                 ret, current_frame = cap.read()
#                 current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
#                 previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
#                 DBF = cv2.absdiff(current_frame_gray, previous_frame_gray)
#                 DOG = current_frame_gray - previous_frame_gray
#                 # score =
#                 # d = psnr(original, contrast)
#                 e = measure.compare_ssim(previous_frame_gray, current_frame_gray)
#                 ssim.append(e)
#                 previous_frame = current_frame.copy()
#                 count += 1
#                 frameindex.append(count)
#                 if count in index:
#                     xa.append(count)
#                     ssima.append(e)
#                     # print(xa)
#                     # print(ssima)
#         else:
#             break
#         plt.figure()
#         plt.plot(frameindex,ssim)
#         plt.scatter(xa, ssima)
#         plt.show()
#     cap.release()
#     cv2.destroyAllWindows()




# for n in range(10):
# # while(cap.isOpened()):
#     current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
#     previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
#
#     frame_diff = cv2.absdiff(current_frame_gray,previous_frame_gray)
#
#     # cv2.imshow('frame diff ',frame_diff)
#     # if cv2.waitKey(1) & 0xFF == ord('q'):
#     #     break
#     cv2.imwrite(os.path.join('./test','%.6d.jpg'%count),current_frame)
#
#     previous_frame = current_frame.copy()
#     ret, current_frame = cap.read()
#     count += 1
# cap.release()
# cv2.destroyAllWindows()

import os
import numpy as np
from numpy.lib.stride_tricks import as_strided


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

# frame1 = cv2.imread('/Users/hjyyyyy/Desktop/code2-master/color-matcher/test/data/000002_6.png')
# frame2 = cv2.imread('/Users/hjyyyyy/Desktop/code2-master/color-matcher/test/data/000002_2.png')
# frame = cv2.imread('/Users/hjyyyyy/Desktop/code2-master/color-matcher/test/data/000002_3.png')
# comp = cv2.imread('/Users/hjyyyyy/Desktop/code2-master/color-matcher/test/data/000002.png')
# frame_diff = cv2.absdiff(frame1,frame2)
# thresh = cv2.threshold(frame_diff[...,0], 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
#
# img = thresh
kernel = np.ones((6,6),np.uint8)
# # erosion = cv2.erode(img,kernel,iterations = 1)
# # dilation = cv2.dilate(img,kernel,iterations = 1)
# # opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
# # closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
# opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
# # closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
# dilation = cv2.dilate(opening,kernel,iterations = 1)
# erosion = cv2.erode(opening,kernel,iterations = 1)
# # gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
# for r in range (720):
#     for c in range(1280):
#         if opening[r,c] == 0:
#             frame[r,c] = comp[r,c]
# cv2.imshow('frame', frame)
# cv2.imwrite('000002_comp.png',frame)
#
# # cv2.imshow('diff',frame_diff)
# # cv2.imshow('closing',closing)
# cv2.imshow('opening',opening)
# # cv2.imshow('dilation',dilation)
# # cv2.imshow('erosion',erosion)
# # cv2.imshow('thresh',thresh)
# cv2.waitKey(0)


current_frame1 = cv2.imread('/Users/hjyyyyy/Desktop/lama3/y/0091_mean_y.png')
current_frame2 = cv2.imread('/Users/hjyyyyy/Desktop/lama3/y/0091_mean_u.png')
# frame = cv2.imread('/Users/hjyyyyy/Desktop/lama3/y/0001_d=0_y.png')
current_frame1 = cv2.cvtColor(current_frame1,cv2.COLOR_BGR2YUV)
current_frame2 = cv2.cvtColor(current_frame2,cv2.COLOR_BGR2YUV)
previous_frame = cv2.imread('/Users/hjyyyyy/Desktop/lama3/yuv420/lama1/0091.png')
previous_frame = cv2.cvtColor(previous_frame,cv2.COLOR_BGR2YUV)
original = cv2.imread('/Users/hjyyyyy/Desktop/lama6/0091.png')
flashframe = cv2.imread('/Users/hjyyyyy/Desktop/lama7/0091.png')
flashframe = cv2.cvtColor(flashframe,cv2.COLOR_BGR2YUV)
interpolation = cv2.imread('/Users/hjyyyyy/Desktop/lama3/yuv420/lama1/0091.png')

comp = previous_frame

Yflash, Uflash, Vflash = cv2.split(flashframe)
Ynoflash, Unoflash, Vnoflash = cv2.split(previous_frame)
diff = cv2.absdiff(original,interpolation)
# current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
frame_diff = current_frame1 - previous_frame
frame_diff1 = cv2.absdiff(current_frame1,previous_frame)
thresh1 = cv2.threshold(frame_diff1[...,0], 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# thresh1 = cv2.threshold(frame_diff1[...,0], 0, 255, cv2.THRESH_TOZERO_INV | cv2.THRESH_OTSU)[1]
frame_diff2 = cv2.absdiff(current_frame2,previous_frame)
thresh2 = cv2.threshold(frame_diff2[...,1], 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# thresh2 = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN, kernel)
# thresh2 = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN, kernel)
# thresh2 = cv2.threshold(frame_diff2[...,1], 0, 255, cv2.THRESH_TOZERO_INV | cv2.THRESH_OTSU)[1]
# thresh = thresh1 - thresh2
thresh = cv2.threshold(diff[...,0], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
cv2.imshow('thresh',thresh)

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

for cc in cnts:
    (x, y, w, h) = cv2.boundingRect(cc)
    d = max(int(w/2), int(h/2))
    i = x + d
    j = y + d
    for r in range(y, y + h):
        for c in range(x, x + w):
            if thresh[r, c] == 255:
                Y_flash = np.reshape(cell_neighbors(Yflash, i, j, d), (-1, 1))
                Y_noflash = np.reshape(cell_neighbors(Ynoflash, i, j, d), (-1, 1))
                regY = LinearRegression().fit(Y_flash, Y_noflash)
                ScaleY = regY.coef_[0]
                OffsetY= regY.intercept_[0]
                U_flash = np.reshape(cell_neighbors(Uflash, i, j, d), (-1, 1))
                U_noflash = np.reshape(cell_neighbors(Unoflash, i, j, d), (-1, 1))
                regU = LinearRegression().fit(U_flash, U_noflash)
                ScaleU = regU.coef_[0]
                OffsetU= regU.intercept_[0]
                V_flash = np.reshape(cell_neighbors(Vflash, i, j, d), (-1, 1))
                V_noflash = np.reshape(cell_neighbors(Vnoflash, i, j, d), (-1, 1))
                regV = LinearRegression().fit(V_flash, V_noflash)
                ScaleV = regV.coef_[0]
                OffsetV= regV.intercept_[0]
                Ynoflash[r, c] = ScaleY * Yflash[r, c] + OffsetY
                Unoflash[r, c] = ScaleU * Uflash[r, c] + OffsetU
                Vnoflash[r, c] = ScaleV * Vflash[r, c] + OffsetV

    cv2.rectangle(interpolation, (x, y), (x + w, y + h), (0, 0, 255), 2)
# cv2.imshow('interpolation',interpolation)


# for r in range (540):
#     for c in range(960):
#         if thresh[r,c] == 255:
#             print(r,c)
#             Ynoflash[r,c] = Yflash[r,c]
#             Unoflash[r, c] = Uflash[r, c]
#             Vnoflash[r, c] = Vflash[r, c]
#             # I_flash = np.reshape(cell_neighbors(Yflash, r, c, d), (-1, 1))
#             # I_noflash = np.reshape(cell_neighbors(Ynoflash, r, c, d), (-1, 1))
#             # regY1 = LinearRegression().fit(I_flash, I_noflash)
#             # ScaleYUV = regY1.coef_[0]
#             # OffsetYUV= regY1.intercept_[0]
#             YUV_flash = np.mean(np.reshape(cell_neighbors(Yflash, r, c, d), (-1, 1)))
#             YUV_noflash = np.mean(np.reshape(cell_neighbors(Ynoflash, r, c, d), (-1, 1)))
#             ScaleYUV = YUV_noflash / YUV_flash
#             OffsetYUV = YUV_noflash - YUV_flash * ScaleYUV
#             comp[r,c,0] =  comp[r,c,0] * ScaleYUV + OffsetYUV
#             # comp[r,c] = flashframe[r,c]
comp[...,0] = Ynoflash
comp[...,1] = Unoflash
comp[...,2] = Vnoflash
comp = cv2.cvtColor(comp,cv2.COLOR_YUV2BGR)
cv2.imshow('comp',comp)
# # cv2.imshow('diff', diff)
# # cv2.imshow('thresh ',thresh)
#
#
# # for i in range(720):
# # #     for j in range(1280):
# # #         if frame_diff[i, j] != 0:
# # #             print(i,j)
# # cv2.imshow('frame diff1 ',frame_diff1)
# # cv2.imshow('frame diff2 ',frame_diff2)
# # cv2.imshow('thresh1 ',thresh1)
# # cv2.imshow('thresh2 ',thresh2)
# cv2.imshow('thresh ',thresh)
# #
# # cv2.imwrite('thresh.png',thresh)
# # cv2.imwrite('diff.png',diff)
# # cv2.imshow('frame diff2 ',frame_diff2)
cv2.waitKey(0)