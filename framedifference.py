import cv2
import os
import math
from skimage import measure
import numpy as np
from matplotlib import pyplot as plt
import csv

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

# current_frame = cv2.imread('/Users/hjyyyyy/Desktop/code2-master/0000modified_d=0.png')
current_frame1 = cv2.imread('/Users/hjyyyyy/Desktop/lama3/y/0001_mean_y.png')
current_frame2 = cv2.imread('/Users/hjyyyyy/Desktop/lama3/y/0001_mean_uv.png')
# frame = cv2.imread('/Users/hjyyyyy/Desktop/lama3/y/0001_d=0027_y.png')
current_frame1 = cv2.cvtColor(current_frame1,cv2.COLOR_BGR2YUV)
current_frame2 = cv2.cvtColor(current_frame2,cv2.COLOR_BGR2YUV)
previous_frame = cv2.imread('/Users/hjyyyyy/Desktop/lama3/yuv420/lama1/0001.png')
previous_frame = cv2.cvtColor(previous_frame,cv2.COLOR_BGR2YUV)
original = cv2.imread('/Users/hjyyyyy/Desktop/render_result/0001.png')
interpolation = cv2.imread('/Users/hjyyyyy/Desktop/lama3/yuv420/lama1/0001.png')
diff = cv2.absdiff(original,interpolation)
cv2.imshow('diff', diff)
# current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
# previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
# frame_diff = current_frame1 - previous_frame
frame_diff1 = cv2.absdiff(current_frame1,previous_frame)
thresh1 = cv2.threshold(frame_diff1[...,0], 1, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
frame_diff2 = cv2.absdiff(current_frame2,previous_frame)
thresh2 = cv2.threshold(frame_diff2[...,0], 1, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
thresh = thresh2 - thresh1
for r in range (540):
    for c in range(960):
        if thresh[r,c] ==255:
            # print('-----')
            print(r,c)
        # if diff[r,c,1] > 0:
        #     print('+++++')
        #     print(r,c)


# for i in range(720):
# #     for j in range(1280):
# #         if frame_diff[i, j] != 0:
# #             print(i,j)
cv2.imshow('frame diff1 ',frame_diff1)
cv2.imshow('frame diff2 ',frame_diff2)
cv2.imshow('thresh1 ',thresh1)
cv2.imshow('thresh2 ',thresh2)
cv2.imshow('thresh ',thresh)
cv2.imwrite('thresh_1.png',thresh)
cv2.imwrite('diff_1.png',diff)
# cv2.imshow('frame diff2 ',frame_diff2)
cv2.waitKey(0)