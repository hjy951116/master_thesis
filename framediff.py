import cv2
import numpy 
import math
previous_frame = cv2.imread('/home/students/jhuang/output/000007.jpg')
current_frame = cv2.imread('/home/students/jhuang/Videocoding/0.png')

current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)    

frame_diff = cv2.absdiff(current_frame_gray,previous_frame_gray)

original = cv2.imread('/home/students/jhuang/output/000007.jpg')
contrast = cv2.imread('/home/students/jhuang/Videocoding/0.png',1)

def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

d=psnr(original,contrast)
print(d)

cv2.imshow('frame diff',frame_diff)
cv2.imwrite('frame_diff_7.png',frame_diff)
     
cv2.waitKey(0) 


