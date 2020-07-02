#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 10:11:26 2020

@author: abel
"""





import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

scalingFactor = 5000.0

FINAL_MASK_STORE='/home/ashfaquekp/output_table/mask/'
MASK_DIR='/home/ashfaquekp/output_table/mrcnn_mask/'
BASE_DIR='/home/ashfaquekp/output_table/'

def color_dom_change(c):
    maxi=np.max(c)
    mini=np.min(c)
    c_final=((c-mini)/(maxi-mini))*255.
    c_final=c_final.astype(np.uint8)
    return c_final

def label_show(label,labels_im):
    contour=np.zeros((480,640))
    contour[labels_im==label]=1.0
    plt.imshow(contour)
    
    
def label_ret(label,labels_im):
    contour=np.zeros((480,640),dtype=np.uint8)
    contour[labels_im==label]=1
    return contour
      


def generate_pointcloud(rgb_file,depth_file,ply_file):
    """
    Generate a colored point cloud in PLY format from a color and a depth image.
    
    Input:
    rgb_file -- filename of color image
    depth_file -- filename of depth image
    ply_file -- filename of ply file
    
    """
    rgb = cv2.imread(rgb_file)
    depth = cv2.imread(depth_file,cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    
    K=np.loadtxt('blend_K_table.txt')
    fx=K[0,0]
    fy=K[1,1]
    cx=K[0,2]
    cy=K[1,2]



    cols=rgb.shape[1]
    rows=rgb.shape[0];
    
    
    coords=np.zeros((480,640,3),dtype=np.float32)
    for u in range(cols):
        for v in range(rows):
            
            Z = depth[v,u] / scalingFactor
            if Z==0: continue
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            coords[v,u,0]=X
            coords[v,u,1]=Y
            coords[v,u,2]=Z
    # print("Coordinated finished...")
    n_map=np.zeros((480,640,3),dtype=np.float32)
    for u in range(cols):
        for v in range(rows):
            if(u==cols-1 or v==rows-1):
                continue
            
            v00=coords[v,u,:]
            v01=coords[v,u+1,:]
            v10=coords[v+1,u,:]
            
            if(v00[2]==0.0 or v01[2]==0.0 or v10[2]==0.0):
                continue
            
            p=v01-v00
            q=v10-v00
            norm=np.cross(p,q)
            norm=norm/np.linalg.norm(norm)
            n_map[v,u,:]=norm
    # print("Normal Map finished....")
    return coords,n_map



if __name__=='__main__':
    file=open(BASE_DIR+'rgb.txt')
    lines = file.read().split("\n")
    print("Number of lines in associate",len(lines))
    for i in tqdm(range(len(lines)-1)):
        
        
        img_path=BASE_DIR+ lines[i].split(" ")[1]
        img_no=lines[i].split(" ")[1].split("_")[1][:4]
        depth_path=BASE_DIR+'depth/'+str(int(img_no))+'.png'
    
        if True:
            vmap,nmap=generate_pointcloud(img_path,depth_path,'/home/abel/blender/bottle_0_norm.ply')
            
        
        
        
        if True:
            # print("Starting to calculate dd and cc vals..")
            cols=vmap.shape[1]
            rows=vmap.shape[0]
            
            dd_vals=np.zeros((480,640))
            cc_vals=np.zeros((480,640))
            for u in range(cols):
                for v in range(rows):
                    if(u==0 or v==0 or u==cols-1 or v==rows-1):
                            continue
                    
                    v_p=vmap[v,u,:]
                    n=nmap[v,u,:]
                    
                    near_vert=[vmap[v-1,u-1,:],vmap[v-1,u,:],vmap[v-1,u+1,:],vmap[v,u-1,:],vmap[v,u+1,:],vmap[v+1,u-1,:],vmap[v+1,u,:],vmap[v+1,u+1,:]]
                    near_norm=[nmap[v-1,u-1,:],nmap[v-1,u,:],nmap[v-1,u+1,:],nmap[v,u-1,:],nmap[v,u+1,:],nmap[v+1,u-1,:],nmap[v+1,u,:],nmap[v+1,u+1,:]]
                    cc_val=np.zeros((8))
                    dd_val=np.zeros((8))
                    
                    for e in range(8):
                        if np.dot((near_vert[e]-v_p),n) <0:
                            cc_val[e]=0
                        else:
                            cc_val[e]=(1-np.dot(near_norm[e],n))
                        dd_val[e]=abs(np.dot((near_vert[e]-v_p),n))   
                    
                    cc_vals[v,u]=np.max(cc_val)
                    dd_vals[v,u]=np.max(dd_val)
            
       
        mask =np.ones((480,640),dtype=np.uint8)
        mask[np.logical_or(dd_vals>0.05,cc_vals>0.3)]=0
        mask=np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        mask=color_dom_change(mask)
    
    
        retval, markers = cv2.connectedComponents(mask[:,:,0],connectivity=8)
        labels,label_counts=np.unique(markers,return_counts=True)
        valid_labels=labels[label_counts>20]
        g_masks=[]
        for k in range(len(valid_labels)):
            img=label_ret(markers,valid_labels[k])
            g_masks.append(img)
       
       
        
        m_masks=[]
        m_mask_id=[]
        file_mask_index=open(MASK_DIR +str(img_no)+'.txt')
        lines_mask = file_mask_index.read().split("\n")

        for l in range(len(lines_mask)-1):
            mfile=lines_mask[l].split(" ")[0]
            mid=lines_mask[l].split(" ")[1]

            m1=cv2.imread(mfile)
            m1[m1<125]=0
            m1[m1>=125]=1
            m_masks.append(m1)
            m_mask_id.append(int(mid))
        
        file_mask_index.close()
        

        
        g_mask_val=np.zeros((len(g_masks))).astype(np.bool)
        g_mask_id=np.zeros((len(g_masks)))
        for l in range(len(g_masks)):
            g_mask_area=g_masks[l].sum()
            g_mask=g_masks[l].astype(np.bool)
            
            best_val=0
            for j in range(len(m_masks)):
                m_mask=m_masks[j].astype(np.bool)
                intersection=(np.logical_and(m_mask[:,:,0],g_mask).astype(np.float)).sum()
                
                val=intersection/g_mask_area
                
                
                if val>0.6 and val>best_val:
                    g_mask_val[l]=True
                    g_mask_id[l]=m_mask_id[j]
                    best_val=max(val,best_val)

        file_txt=open(FINAL_MASK_STORE+str(img_no)+'.txt','w')         
        for l in range(len(g_masks)):
            if g_mask_val[l]==False:
                continue
            mask_img=g_masks[l]
            mask_img=mask_img*255
            mask_img=mask_img.astype('uint8')
            #print("Class of ",i,":",class_names[class_ids[i]])
            rand_n = random.randint(0,999)
            mask_file_name=FINAL_MASK_STORE+str(img_no)+'_'+str(int(g_mask_id[l]))+'_'+str(rand_n)+'.png'
            cv2.imwrite(mask_file_name,mask_img)
            file_txt.write("%d %s \n"%(int(g_mask_id[l]),mask_file_name))
        file_txt.close()
        
    file.close()