import math
import numpy as np
from PIL import Image
import time

def compute_h(p1, p2):
    # p1=H*p2
    n = p1.shape[0]
    A = np.zeros((2*n,9))

    for i in range(n):
        A[2*i,0] = p2[i,0]
        A[2*i,1] = p2[i,1]
        A[2*i,2] = 1
        A[2*i,6] = -p1[i,0]*p2[i,0]
        A[2*i,7] = -p1[i,0]*p2[i,1]
        A[2*i,8] = -p1[i,0]         
        
        A[2*i+1,3] = p2[i,0]
        A[2*i+1,4] = p2[i,1]
        A[2*i+1,5] = 1         
        A[2*i+1,6] = -p1[i,1]*p2[i,0]
        A[2*i+1,7] = -p1[i,1]*p2[i,1]
        A[2*i+1,8] = -p1[i,1]
    
    ATA = np.dot(np.transpose(A),A)
    U, S, V = np.linalg.svd(ATA)
    unit_t = U[:,-1]
    H = np.reshape(unit_t,(3,3))

    return H

def compute_h_norm(p1, p2):

    p1_norm = p1/1600
    p2_norm = p2/1600
    N = [1/1600,1/1600,1,1/1600,1/1600,1,1/(1600*1600),1/(1600*1600),1/1600]
    H = compute_h(p1_norm,p2_norm)
    H = np.reshape(np.dot(np.diag(N),np.reshape(H,(9,1))),(3,3))
    
    return H

def warp_image(igs_in, igs_ref, H):
    
    source = igs_in.copy()
    reference = igs_ref.copy()
    h = H/H[2,2]
    inv_H = np.linalg.inv(h)
    
    coor_00 = np.dot(h,[0,0,1])
    coor_0w = np.dot(h,[1599,0,1]) 
    coor_h0 = np.dot(h,[0,1199,1]) 
    coor_hw = np.dot(h,[1599,1199,1])
    max_x = max(coor_0w[0]/coor_0w[2],coor_hw[0]/coor_hw[2])
    min_x = min(coor_00[0]/coor_00[2],coor_h0[0]/coor_h0[2])
    max_y = max(coor_h0[1]/coor_h0[2],coor_hw[1]/coor_hw[2])
    min_y = min(coor_00[1]/coor_00[2],coor_0w[1]/coor_0w[2])
    
    orix = math.floor(coor_00[0]/coor_00[2])
    oriy = math.floor(coor_00[1]/coor_00[2])
    
    igs_warp = np.zeros((math.ceil(max_y)-math.floor(min_y),math.ceil(1600-math.floor(min_x)),3))
    
    for x in range(igs_warp.shape[1]):
        for y in range(igs_warp.shape[0]):
            ref = np.array([[x+orix],[y+oriy],[1]])
            res = np.dot(inv_H,ref)
            x_ = res[0]/res[2]
            y_ = res[1]/res[2]
            if x_ >= 0 and x_ < source.shape[1]-1 and y_ >= 0 and y_ < source.shape[0]-1:
                x_floor = math.floor(x_)
                y_floor = math.floor(y_)                
                a = x_ - x_floor
                b = y_ - y_floor
                term1 = (1-b)*(1-a)*source[y_floor,x_floor,:]
                term2 = b*(1-a)*source[y_floor+1,x_floor,:]
                term3 = (1-b)*a*source[y_floor,x_floor+1,:]
                term4 = b*a*source[y_floor+1,x_floor+1,:]
                igs_warp[y,x,:] = term1+term2+term3+term4
            else:
                igs_warp[y,x,:] = 0.1
    print("warping done")
    ref_merge = np.zeros(igs_warp.shape)
    ref_merge[-oriy:1200-oriy,-orix:1600-orix,:] = reference
    
    igs_merge = igs_warp.copy()
    for ch in range(3):
        for i in range(igs_merge.shape[0]):
            for j in range(igs_merge.shape[1]):
                if(igs_merge[i,j,ch] == 0.1 ):
                    igs_merge[i,j,ch] = ref_merge[i,j,ch]
    print("merging done")
    return igs_warp[-oriy:1200-oriy,-orix:1600-orix,:], igs_merge

def rectify(igs, p1, p2):
    H = compute_h_norm(p2,p1)
    inv_h = np.linalg.inv(H/H[2,2])
    igs_rec = igs.copy()
    
    
    for y in range(igs_rec.shape[0]):
        for x in range(igs_rec.shape[1]):
            res = np.dot(inv_h,[x,y,1])
            x_ = res[0]/res[2]
            y_ = res[1]/res[2]
            if x_>=0 and x_<igs_rec.shape[1]-1 and y_>=0 and y_<igs_rec.shape[0]-1:
                flx = math.floor(x_)
                fly = math.floor(y_)
                a = x_ - flx
                b = y_ - fly
                term1 = (1-b)*(1-a)*igs[fly,flx,:]
                term2 = b*(1-a)*igs[fly+1,flx,:]
                term3 = (1-b)*a*igs[fly,flx+1,:]
                term4 = b*a*igs[fly+1,flx+1,:]
                igs_rec[y,x,:] = term1+term2+term3+term4
            else:
                igs_rec[y,x,:] = 0               
        
    return igs_rec

def set_cor_mosaic():
    p_in = np.array([[1282,417],[1443,405],[1294,510],[1446,505],[845,862],[943,864],[1068,555],[1255,959]])
    p_ref = np.array([[535,424],[676,423],[537,513],[679,512],[58,896],[181,888],[320,553],[509,948]])
                    

    return p_in, p_ref

def set_cor_rec():
    c_in = np.array([[1069,179],[1382,139],[1058,849],[1381,861]])
    c_ref = np.array([[600,150],[900,150],[600,770],[900,770]])
    
    return c_in, c_ref


def main():
    ##############
    # step 1: mosaicing
    ##############

    # read images
    img_in = Image.open('data/porto1.png').convert('RGB')
    img_ref = Image.open('data/porto2.png').convert('RGB')

    # shape of igs_in, igs_ref: [y, x, 3]
    igs_in = np.array(img_in)
    igs_ref = np.array(img_ref)

    # lists of the corresponding points (x,y)
    # shape of p_in, p_ref: [N, 2]
    p_in, p_ref = set_cor_mosaic()

    # p_ref = H * p_in
    H = compute_h_norm(p_ref, p_in)
    print("warping start")
    igs_warp, igs_merge = warp_image(igs_in, igs_ref, H)

    # plot images
    img_warp = Image.fromarray(igs_warp.astype(np.uint8))
    img_merge = Image.fromarray(igs_merge.astype(np.uint8))

    # save images
    img_warp.save('porto1_warped.png')
    img_merge.save('porto_mergeed.png')

    ##############
    # step 2: rectification
    ##############

    img_rec = Image.open('data/iphone.png').convert('RGB')
    igs_rec = np.array(img_rec)

    c_in, c_ref = set_cor_rec()
    print("rectify start")
    igs_rec = rectify(igs_rec, c_in, c_ref)
    print("rectify end")
    img_rec = Image.fromarray(igs_rec.astype(np.uint8))
    img_rec.save('iphone_rectified.png')


if __name__ == '__main__':
    start = time.time()
    main()
    print("time :", time.time() - start)
