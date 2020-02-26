import numpy as np
import cv2


def flash_no_flash(img1, img2, d, sig_col, sigma):
    flash = cv2.imread(img1, 0)
    no_flash = cv2.imread(img2, 0)

    if d == -1:
        d = 9
    if sig_col == -1:
        sig_col = 30
    if sigma == -1:
        sigma = (((flash.shape[0]**2)+(flash.shape[1]**2))**0.5)*0.025

    base_f = cv2.bilateralFilter(flash, d, sig_col, sigma)
    base_nf = cv2.bilateralFilter(no_flash, d, sig_col, sigma)

    flash = flash.astype('float')
    base_f = base_f.astype('float')
    detail = cv2.divide(flash, base_f)

    base_nf = base_nf.astype('float')
    intensity = cv2.multiply(base_nf, detail)

    no_flash = no_flash.astype('float')
    nflash_color = cv2.imread(img2, 1)
    nflash_color = nflash_color.astype('float')
    b = nflash_color[:, :, 0]
    g = nflash_color[:, :, 1]
    r = nflash_color[:, :, 2]
    b = cv2.divide(b, no_flash)
    g = cv2.divide(g, no_flash)
    r = cv2.divide(r, no_flash)

    intensity=intensity.astype('float')
    b = cv2.multiply(b, intensity)
    g = cv2.multiply(g, intensity)
    r = cv2.multiply(r, intensity)

    result = np.zeros((flash.shape[0], flash.shape[1],3), np.uint8)
    result[:, :, 0] = b
    result[:, :, 1] = g
    result[:, :, 2] = r

    cv2.imshow('result',result)
    frame_diff = cv2.absdiff(cv2.cvtColor(cv2.imread(img1), cv2.COLOR_BGR2GRAY),cv2.cvtColor(result, cv2.COLOR_BGR2GRAY))
    # cv2.imshow('framediff',frame_diff)
    cv2.waitKey(1)

    cv2.imwrite(os.path.join('/home/students/jhuang/result','%.6dr.png'%index[i]), result)

if __name__ == "__main__":
    path1 = '/home/students/jhuang/crew_A/'
    fileList1 = os.listdir(path1)
    fileList1.sort()
    path2 = '/home/students/jhuang/crew_B/'
    fileList2 = os.listdir(path2)
    fileList2.sort()
    for i in range (len(index)-1):
        img1 = path1 + os.sep + fileList1[i]
        print(img1)
        cv2.imshow('img1',cv2.imread(img1))
        img2 = path2 + os.sep + fileList2[i]
        cv2.imshow('img2',cv2.imread(img2))
        flash_no_flash(img1,img2,9,30,-1)

