import os
import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline
from skimage.filters import apply_hysteresis_threshold
import time
from matplotlib import pyplot as plt


def lucas_kanade_affine(img1, img2, p, Gx, Gy):
    ### START CODE HERE ###
    temp = img1.copy()
    img  = img2.copy()/255.0
    Gx_ = Gx.copy()/255.0
    Gy_ = Gy.copy()/255.0

    height, width = temp.shape
    x = np.arange(0,width,1)
    y = np.arange(0,height,1)
    img_spline = RectBivariateSpline(y, x, img)
    Gx_spline  = RectBivariateSpline(y, x, Gx_)
    Gy_spline  = RectBivariateSpline(y, x, Gy_)
    gxmean = np.mean(np.abs(Gx))
    gymean = np.mean(np.abs(Gy))
    
    H = np.zeros((6,6))
    error_sum = np.zeros((6,1))
    count = 0
    total = len(x)*len(y)
    for y_ in y:
        for x_ in x:
            wx = (1 + p[0,0])*x_ + p[2,0]*y_ + p[4,0]
            wy = p[1,0]*x_ + (1 + p[3,0])*y_ + p[5,0]
            if (wx >= 0 and wy >= 0 and wx <= width-1 and wy <= height-1):
                gx = float(Gx_spline.ev(wy,wx)*255.0)
                gy = float(Gy_spline.ev(wy,wx)*255.0)
                if (abs(gx) < gxmean and abs(gy) < gymean): 
                    count += 1
                    continue
                value = float(img_spline.ev(wy,wx)*255.0)
                if (value < 0 or value > 255):
                    continue
                error = temp[y_,x_] - value
                grad = np.array([[gx*x_],
                                 [gy*x_],
                                 [gx*y_],
                                 [gy*y_],
                                 [gx],
                                 [gy]])
                error_sum = error_sum + grad*error
                H = H + grad@grad.T
    print(count,"/",total)
    inv_H = np.linalg.pinv(H)
    dp = inv_H@error_sum    
    ### END CODE HERE ###
    return dp

def subtract_dominant_motion(img1, img2):
    Gx = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize = 5) # do not modify this
    Gy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize = 5) # do not modify this
    th_hi = 0.4*256 # you can modify this
    th_lo = 0.25*256 # you can modify this

    ### START CODE HERE ###
    P = np.zeros((6,1))
    
    temp = img1.copy()
    img  = img2.copy()
    dp = lucas_kanade_affine(temp, img, P, Gx, Gy)
    P = dp
    first = np.linalg.norm(dp)
    print(first)

    height, width = img.shape
    x = np.arange(0,width,1)
    y = np.arange(0,height,1)
    temp_norm = temp/255.0
    temp_spline = RectBivariateSpline(y, x, temp_norm)
    
    moving_image = np.zeros(img.shape)
    for y_ in y:
        for x_ in x:
            wx = ((x_-P[4,0])*(1+P[3,0])-(y_-P[5,0])*P[2,0])/(-P[1,0]*P[2,0]+(1+P[0,0])*(1+P[3,0]))
            wy = ((x_-P[4,0])*P[1,0]-(y_-P[5,0])*(1+P[0,0]))/(P[1,0]*P[2,0]-(1+P[0,0])*(1+P[3,0]))
            if (wx >= 0 and wy >= 0 and wx <= width-1 and wy <= height-1):
                value = float(temp_spline.ev(wy,wx)*255)
                if (value >= 0 and value <= 255):
                    moving_image[y_][x_] = abs(img[y_][x_] - value)
    
    ### END CODE HERE ###
    hyst = apply_hysteresis_threshold(moving_image, th_lo, th_hi)
    plt.figure()
    plt.imshow(hyst,cmap='gray')
    plt.axis('off')
    plt.show()
    return hyst

if __name__ == "__main__":
    start = time.time()
    data_dir = 'data'
    video_path = 'motion.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 150/20, (636, 318))
    tmp_path = os.path.join(data_dir, "organized-{}.jpg".format(0))
    T = cv2.cvtColor(cv2.imread(tmp_path), cv2.COLOR_BGR2GRAY)
    for i in range(0, 50):
        img_path = os.path.join(data_dir, "organized-{}.jpg".format(i))
        I = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
        clone = I.copy()
        moving_img = subtract_dominant_motion(T, I)
        clone = cv2.cvtColor(clone, cv2.COLOR_GRAY2BGR)
        clone[moving_img, 2] = 522
        out.write(clone)
        T = I
    out.release()
    print((time.time()-start)/60)
    
