import math
import glob
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

# parameters
datadir = './data'
resultdir='./results'

# edgedetection
Gaussian_size= (5,5)
sigmalist=[1.0,3.0,2.0,2.0]
highThresholdlist=[0.2,0.29,0.32,0.24] #max에서 몇퍼센트
lowThresholdlist=[0.01,0.5,0.1,0.1] #highThreshold에서 몇퍼센트

#houghline
rhoReslist=[10,20,3,5]
thetaReslist=[20,14,22.5,22.5] 
intervallist=[5,10,2,8]
nLines=20


def replic_pad(Igs,size):
    H,W = Igs.shape
    h,w = size 
    h2,w2 = h//2, w//2
    padded = np.zeros((H+2*h2,W+2*w2))
    padded[h2:H+h2,w2:W+w2] = Igs[:,:].copy()
    
    for i in range(0,h2):
        padded[i,:] = padded[h2,:]
        padded[H+2*h2-1-i,:] = padded[H+h2-1,:]

    for j in range(0,w2):
        padded[:,j] = padded[:,w2].copy()
        padded[:,W+2*w2-1-j] = padded[:,W+w2-1]       
    return padded  
    

def ConvFilter(Igs, G):
    for s in G.shape:
        if s % 2 == 0:
            raise Exception("Kernel size must be odd")
    
    h, w = G.shape 
    h2, w2 = h//2, w//2
    padded = replic_pad(Igs.copy(), G.shape)
    H, W = padded.shape 
        
    #flip kernel
    for i in range(h2):
        G[i,:], G[2*h2-i,:] = G[2*h2-i,:].copy(), G[i,:].copy()
    for j in range(w2):
        G[:,j], G[:,2*w2-j] = G[:,2*w2-j].copy(), G[:,j].copy()
       
    #convolve
    Iconv = np.zeros(Igs.shape)   
    for i in range(h2,H-h2):
        for j in range(w2,W-w2):
            sum = 0
            for k in range(h):
                for l in range(w):
                    sum += padded[i-h2+k,j-w2+l]*G[k,l]
            Iconv[i-h2,j-w2] = sum
    
    return Iconv

def GaussianFilter(Igs,sigma,size):
    
    h,w = size
    h2,w2 = h//2, w//2
    
    col = np.zeros((h,1))
    row = np.zeros((1,w))
    colsum, rowsum = 0, 0
    col1 = math.exp(-math.pow(-h2,2)/(2*math.pow(sigma,2)))/(math.sqrt(2*math.pi)*sigma)
    row1 = math.exp(-math.pow(-w2,2)/(2*math.pow(sigma,2)))/(math.sqrt(2*math.pi)*sigma)
   
    for i in range(h):
        col[i,:] = math.exp(-math.pow(i-h2,2)/(2*math.pow(sigma,2)))/(math.sqrt(2*math.pi)*sigma)
        col[i,:] = np.round(col[i,:]/col1)
        colsum += col[i,:]       
    for j in range(w):
        row[:,j] = math.exp(-math.pow(j-w2,2)/(2*math.pow(sigma,2)))/(math.sqrt(2*math.pi)*sigma)
        row[:,j] = np.round(row[:,j]/row1) 
        rowsum += row[:,j]

    gaussian = np.zeros(Igs.shape)
    gaussian = ConvFilter(ConvFilter(Igs.copy(),col),row)

    return gaussian/(rowsum*colsum)

def NMS(Im,Io):
    
    H,W = Im.shape
    nms = Im.copy()
       
    for i in range(0,H):
        for j in range(0,W):
            if (i !=0 and j !=0 and i != H-1 and j != W-1):
                if (Io[i,j] < 0 ):
                    Io[i,j] += 180
                if (0 <= Io[i,j] < 22.5 or 157.5 <= Io[i,j] <= 180):
                    if(nms[i ,j] < nms[i,j+1] or nms[i,j] < nms[i,j-1]):
                        nms[i,j] = 0
                elif (22.5 <= Io[i,j] < 67.5):
                    if(nms[i,j] < nms[i-1,j+1] or nms[i,j] < nms[i+1,j-1]):
                        nms[i,j] = 0
                elif (67.5 <= Io[i,j] < 112.5):
                    if(nms[i,j] < nms[i-1,j] or nms[i,j] < nms[i+1,j]):
                        nms[i,j] = 0
                elif (112.5 <= Io[i,j] < 157.5):
                    if(nms[i,j] < nms[i-1,j-1] or nms[i,j] < nms[i+1,j+1]):
                        nms[i,j] = 0
            else:
                nms[i,j] = 0
    
    return nms

            
def doubleTH(Im, highThreshold, lowThreshold):
    
    H,W = Im.shape
    highTH = Im.max()*highThreshold
    lowTH = highTH*lowThreshold
    TH = np.zeros(Im.shape)
    strong_edge = 1
    weak_edge = 0.5
    str_num = 0
    weak_num = 0
    
    #double Threshold
    for i in range(H):
        for j in range(W):
            if (Im[i,j] >= highTH):
                TH[i,j] = strong_edge
                str_num += 1 
            elif (Im[i,j] < highTH and Im[i,j] >= lowTH):
                TH[i,j] = weak_edge
                weak_num += 1 
    print("총", H*W,"strong",str_num,"weak",weak_num)
    
    str_num2 = 0
    #Hysteresis based
    hys = TH.copy()           
    for i in range(0,H):
        for j in range(0,W):
            if (i!=0 and i!=H-1 and j!=0 and j!=W-1):
                if (TH[i,j] == weak_edge):
                    if ((TH[i,j-1] == strong_edge) or
                        (TH[i-1,j-1] == strong_edge) or
                        (TH[i-1,j] == strong_edge) or
                        (TH[i-1,j+1] == strong_edge) or
                        (TH[i,j+1] == strong_edge) or
                        (TH[i+1,j+1] == strong_edge) or
                        (TH[i+1,j] == strong_edge) or
                        (TH[i+1,j-1] == strong_edge)):
                        hys[i,j] = strong_edge
                        str_num2 += 1
                    else:
                        hys[i,j] = 0
            else:
                hys[i,j] = 0
    print(weak_num,"중",str_num2,"개 strong에 추가")
                    
    return hys
                
    
    
def EdgeDetection(Igs, sigma, highThreshold, lowThreshold):
    
    sblx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]],np.float32)
    sbly = np.array([[-1,-2,-1],[0,0,0],[1,2,1]],np.float32)
    
    smoothed = GaussianFilter(Igs.copy(),sigma,Gaussian_size)

    Ix = ConvFilter(smoothed,sblx)
    Iy = ConvFilter(smoothed,sbly)
    
    Io = np.zeros(smoothed.shape)
    Io = np.arctan2(Iy,Ix)*180/math.pi
    
    Im = np.zeros(smoothed.shape)
    for i in range(smoothed.shape[0]):
        for j in range(smoothed.shape[1]):
            Im[i,j] = math.sqrt(math.pow(Ix[i,j],2)+math.pow(Iy[i,j],2))
    Im = Im/Im.max()
    
    Im = NMS(Im,Io)
    Im = doubleTH(Im,highThreshold,lowThreshold)
    
    return Im, Io, Ix, Iy

def HoughTransform(Im, rhoRes, thetaRes):
    
    H,W = Im.shape    
    rho_max = math.ceil(math.sqrt(H*H + W*W))
    theta_max = 360                       
    rho = np.arange(0,rho_max,rhoRes)
    theta = np.arange(0,theta_max, thetaRes)
    
    cos_ = np.cos(theta*math.pi/180)
    sin_ = np.sin(theta*math.pi/180)
    rho_len = len(rho)
    theta_len = len(theta)

    acc = np.zeros((rho_len,theta_len))
    for i in range(H):
        for j in range(W):
            if (Im[i,j] != 0):
                for theta_index in range(theta_len):
                    rho_ = round(j*cos_[theta_index]+i*sin_[theta_index]) #실제 rho 값
                    if (0<= rho_ <=rho_max):
                        acc[rho_//rhoRes,theta_index] += 1

    return acc

def HoughLines(H,rhoRes,thetaRes,nLines):

    rth = 1
    tth = 1
    h,w = H.shape
    nonzero = np.transpose(np.nonzero(H))
    theta_max = 360
    theta = np.arange(0,theta_max,thetaRes) #실제 theta
    
    for nz in nonzero:
        U = nz[0]-rth
        D = nz[0]+rth+1        
        L = nz[1]-tth
        R = nz[1]+tth+1
        c = np.arange(L,R,1)
        if (U < 0):
            U = 0
        if (D > h):
            D = h            
        if (R > w):
            c = list(range(L,w,1))+list(range(0,R-w,1))
        H[nz[0],nz[1]] -= 1    
        if (np.any(H[U:D,c]>H[nz[0],nz[1]])):
            H[nz[0],nz[1]] = 0
        else:
            H[nz[0],nz[1]] += 1
                
    IRho, ITheta = np.unravel_index(np.argsort(-1*H,axis=None)[:nLines],H.shape) #rho, theta index 값들 array
    
    for i in range(len(IRho)):
        print(i+1,"번째로 큰 vote 수:", H[IRho[i],ITheta[i]],"(",IRho[i]*rhoRes,theta[ITheta[i]],")")
    return IRho,ITheta

def HoughLineSegments(IRho: list, ITheta: list, Im, thetaRes, rhoRes, interval):
    l = []      
    theta_max = 360
    interval = interval
    
    theta = np.arange(0,theta_max,thetaRes)
    nonzero = np.transpose(np.nonzero(Im))    
    cos_ = np.cos(theta*math.pi/180)
    sin_ = np.sin(theta*math.pi/180)  

    for line_num in range(len(IRho)):
        
        theta_ = ITheta[line_num] #theta index 값
        rho_ = IRho[line_num] #rho index 값
        point_list = [] #일단 edge 위의 모든 점을 point_list에 저장
        segment = []
        temp = []
        
        #기여한 x,y 좌표 찾기
        for point in nonzero:
                    if(rho_ == round(point[1]*cos_[theta_]+point[0]*sin_[theta_])//rhoRes):
                        point_list.append(point)
        point_list = sorted(point_list, key=lambda point_list: point_list[1]) #x값 기준으로 sort
        
        #segment 나누기
        for i in range(len(point_list)):
            if (i>0):
                if(abs(point_list[i][1] - point_list[i-1][1]) > interval or abs(point_list[i][0]-point_list[i-1][0]) > interval):
                    segment.append(temp)
                    temp = []
            temp.append(point_list[i])
        segment.append(temp)
        #print("segment 개수", len(segment))
               
        #가장 긴 segment 찾기
        maxlist = [] #max(segment,key=len)
        max_len = 0 #최소 길이     
        for line in segment:
            if (len(line) < 2): continue
            line_length = math.sqrt(math.pow(line[0][0]-line[-1][0],2)+math.pow(line[0][1]-line[-1][1],2))
            if (line_length >= max_len):
                max_len = line_length
                maxlist = line
                
        if(len(maxlist)!=0):
            d = {'start':maxlist[0],'end':maxlist[-1]}
            l.append(d)
            #print("직선",line_num+1,":",maxlist[0],maxlist[-1])
      
    return l

def drawhoughline(Igs,lRho,lTheta, thetaRes, rhoRes):
    for i in range(len(lRho)):
        theta_ = lTheta[i]*thetaRes*math.pi/180
        rho_ = lRho[i]*rhoRes
        line = []
        if (theta_ > 0):
            for x in range(1,Igs.shape[1]):
                y = -x/np.tan(theta_) + rho_/np.sin(theta_)
                if y >=0 and y < Igs.shape[0]:
                    line.append((x,y))
        elif (theta_ == 0):
            for y in range(1,Igs.shape[0]):
                x = rho_
                if x >= 0 and x < Igs.shape[1]:
                    line.append((x,y))
        if len(line) > 0:
            #print([line[0][0],line[-1][0]],[line[0][-1],line[-1][-1]])
            plt.plot([line[0][0],line[-1][0]],[line[0][-1],line[-1][-1]])
        elif len(line) == 0:
            print(i+1, "번째 IRho의 element에서는 line list가 비어잇음")   
    return

def drawsegments(I):
    for i in range(len(I)):
        plt.plot([I[i]['start'][1],I[i]['end'][1]],[I[i]['start'][0],I[i]['end'][0]],linewidth=2, marker=".")
    return

def main():   
    # read images
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)
    image_number = 0
    for img_path in glob.glob(datadir+'/*.jpg'):
        if image_number == -1: 
            image_number += 1
            continue
        # edgedetection
        sigma=sigmalist[image_number]
        highTh=highThresholdlist[image_number] #max에서 몇퍼센트
        lowTh=lowThresholdlist[image_number]   #highThreshold에서 몇퍼센트
        #houghline
        rhoRes=rhoReslist[image_number]
        thetaRes=thetaReslist[image_number] 
        interval=intervallist[image_number]

        
        # load grayscale image
        origin = Image.open(img_path).convert("RGB")
        img = Image.open(img_path).convert("L")
        Igs = np.array(img)
        Igs = Igs / 255.

        # Hough function
        Im, Io, Ix, Iy = EdgeDetection(Igs.copy(), sigma, highTh, lowTh)
        H= HoughTransform(Im.copy(), rhoRes, thetaRes)
        lRho,lTheta =HoughLines(H.copy(),rhoRes,thetaRes,nLines)
        l = HoughLineSegments(lRho.copy(), lTheta.copy(), Im.copy(),thetaRes, rhoRes,interval)

        # saves the outputs to files
        # Im, H, Im + hough line , Im + hough line segments
        
        plt.figure()
        plt.imshow(Im,cmap='gray')
        plt.axis('off')
        plt.savefig(resultdir +'/img'+ str(image_number+1) +'_Im.jpeg')
        plt.show()

        plt.figure()
        plt.imshow(H,cmap='gray')
        plt.axis('on')
        plt.savefig(resultdir +'/img'+ str(image_number+1) +'_H.jpeg')
        plt.show()

        plt.figure()
        plt.imshow(origin)
        plt.axis('off')
        drawhoughline(Igs,lRho,lTheta, thetaRes, rhoRes)
        plt.savefig(resultdir +'/img'+ str(image_number+1) +'_Im + houghline.jpeg')
        plt.show()

        plt.figure()
        plt.imshow(origin)
        plt.axis('off')
        drawsegments(l)
        plt.savefig(resultdir +'/img'+ str(image_number+1) +'_Im + houghtlinesegments.jpeg')
        plt.show()  
        
        image_number += 1
    

if __name__ == '__main__':
    main()
