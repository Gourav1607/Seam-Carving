#!/usr/bin/env python
# coding: utf-8

# Gourav Siddhad
# 20911004
# g_siddhad@cs.iitr.ac.in

#################################################################

import numpy as np
import cv2
import matplotlib.pyplot as plt

#################################################################

# image energy function using abs of gradients
def compute_energy(img):
    gx = np.gradient(img, axis=1)
    gy = np.gradient(img, axis=0)
    energy = np.abs(gx) + np.abs(gy)
    return energy

def compute_cumulative_min_energy_vertical(energy):
    energy = compute_cumulative_min_energy_horizontal(energy.T)
    return energy.T

def compute_cumulative_min_energy_horizontal(energy):
    r,c = energy.shape
    cmenergy = np.zeros(energy.shape)
    cmenergy[r-1, :] = energy[r-1, :]
    
    for i in range(r-2, -1, -1):
        for j in range(c):
            c1, c2 = max(j-1, 0), min(c, j+2)
            cmenergy[i][j] = energy[i][j] + cmenergy[i+1, c1:c2].min()
                
    return cmenergy

def compute_optimal_seam_vertical(cumulative_energy_v):
    vseam = compute_optimal_seam_horizontal(cumulative_energy_v.T)
    return vseam

def compute_optimal_seam_horizontal(cumulative_energy_h):
    ceh = cumulative_energy_h
    r, c = ceh.shape
    
    hseam = []
    j = ceh[0].argmin()
    hseam.append(j)
    
    for i in range(r-1):
        c1, c2 = max(j-1, 0), min(c, j+2)
        j = max(j-1, 0) + ceh[i+1, c1:c2].argmin()
        hseam.append(j)

    return hseam

def remove_seam_vertical(img, seam_v):
    newImg = remove_seam_horizontal(img.T, seam_v)
    return newImg.T

def remove_seam_horizontal(img, seam_h):
    r, c = img.shape
    newImg = np.zeros((r, c))
    for i, j in enumerate(seam_h):
        newImg[i, 0:j] = img[i, 0:j]
        newImg[i, j:c-1] = img[i, j+1:c]
    return newImg[:, :-1].astype(np.int32)

#################################################################

def sc_resize_image(img, nshape):
    img_out = img.copy()
    
    vdiff = min(img.shape[0] - nshape[0], img.shape[0])
    hdiff = min(img.shape[1] - nshape[1], img.shape[1])
    
    for i in range(vdiff):
        print('Removing VSeams - {:03d}/{:03d} - ({:03d},{:03d})'.format(i+1, vdiff, img_out.shape[0], img_out.shape[1]), end='\r')

        energy = compute_energy(img_out)
        cev = compute_cumulative_min_energy_vertical(energy)
        seam_v = compute_optimal_seam_vertical(cev)
        img_out = remove_seam_vertical(img_out, seam_v)
        if i==0:
            bkp_e, bkp_vc = energy, cev
            display_seam(img, seam_v, True)
    
    for j in range(hdiff):
        print('Removing HSeams - {:03d}/{:03d} - ({:03d},{:03d})'.format(j+1, hdiff, img_out.shape[0], img_out.shape[1]), end='\r')

        energy = compute_energy(img_out)
        ceh = compute_cumulative_min_energy_horizontal(energy)
        seam_h = compute_optimal_seam_horizontal(ceh)
        img_out = remove_seam_horizontal(img_out, seam_h)
        if j==0:
            bkp_hc = ceh
            display_seam(img, seam_v, False)
            
    if vdiff <= 0:
        bkp_vc = np.zeros(img.shape)
    if hdiff <= 0:
        bkp_hc = np.zeros(img.shape)
        
    return bkp_e, bkp_vc, bkp_hc, img_out

#################################################################

def sc_resize_energy(img, nshape):
    img_out = img.copy()
    
    vdiff = min(img.shape[0] - nshape[0], img.shape[0])
    hdiff = min(img.shape[1] - nshape[1], img.shape[1])
    
    for i in range(vdiff):
        print('Removing VSeams - {:03d}/{:03d} - ({:03d},{:03d})'.format(i+1, vdiff, img_out.shape[0], img_out.shape[1]), end='\r')

        cev = compute_cumulative_min_energy_vertical(img_out)
        seam_v = compute_optimal_seam_vertical(cev)
        img_out = remove_seam_vertical(img_out, seam_v)
        if i==0:
            bkp_vc = cev
            display_seam(img, seam_v, True)
    
    for j in range(hdiff):
        print('Removing HSeams - {:03d}/{:03d} - ({:03d},{:03d})'.format(j+1, hdiff, img_out.shape[0], img_out.shape[1]), end='\r')

        ceh = compute_cumulative_min_energy_horizontal(img_out)
        seam_h = compute_optimal_seam_horizontal(ceh)
        img_out = remove_seam_horizontal(img_out, seam_h)
        if j==0:
            bkp_hc = ceh
            display_seam(img, seam_h, False)
    
    if vdiff <= 0:
        bkp_vc = np.zeros(img.shape)
    if hdiff <= 0:
        bkp_hc = np.zeros(img.shape)
    
    return bkp_vc, bkp_hc, img_out

#################################################################

def read_energy(fname):
    energy = []
    with open(fname, 'r') as fp: 
        while True: 
            line = fp.readline() 
            if not line: 
                break
            energy.append(line.strip().split())
    return np.array(energy, dtype='float32')

def write_energy(fname, energy):
    try:
        with open('o_' + fname, 'w') as wp:
            for en in energy:
                for e in en:
                    e = '{:2.2f}    '.format(e)
                    wp.write(e)
                wp.write('\n')
    except:
        print('Error while writing file')

#################################################################

def display_seam(img, seam, vflag=True):
    newimg = np.zeros((img.shape[0], img.shape[1], 3))
    newimg[:, :, 0] = img
    for i, pt in enumerate(seam):
        print(i, end='\r')
        if vflag:
            newimg = cv2.circle(newimg, (i, pt), radius=1, color=(0, 0, 255), thickness=1)
        else:
            newimg = cv2.circle(newimg, (pt, i), radius=1, color=(0, 0, 255), thickness=1)
            
    wimg = cv2.cvtColor(np.array(newimg, dtype='uint8'), cv2.COLOR_BGR2GRAY)
    write_energy('display_seam.txt', wimg)
    
    plt.imshow(cv2.cvtColor(np.array(newimg, dtype='uint8'), cv2.COLOR_BGR2RGB))
    if vflag:
        plt.savefig('display_seam_v.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
    else:
        plt.savefig('display_seam_h.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.show()

#################################################################

# imname = 'img_energy.txt'
imname = input('Enter Energy Text File (with directory) : ')
energy = read_energy(imname)

plt.imshow(energy)
plt.show()

print('The energy image has a resolution of (X Y) : ', energy.shape[0], energy.shape[1])
nshape = input('Enter New Resolution for Image (X and Y seperated by space) : ')
nshape = nshape.split()
nshape = np.array(nshape, dtype='int32')

# Driver Function
bkp_vc, bkp_hc, img_out = sc_resize_energy(energy, nshape)

plt.imshow(energy)
plt.savefig('01_Energy.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
plt.show()

plt.imshow(bkp_vc)
plt.savefig('02_VEnergy.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
plt.show()

plt.imshow(bkp_hc)
plt.savefig('03_HEnergy.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
plt.show()

plt.imshow(img_out)
plt.savefig('04_Output.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
plt.show()

write_energy(imname, img_out)

fig, ax = plt.subplots(1, 4, figsize=(15, 15))

ax[0].imshow(energy)
ax[0].set_title('Energy')

ax[1].imshow(bkp_vc)
ax[1].set_title('VEnergy')

ax[2].imshow(bkp_hc)
ax[2].set_title('HEnergy')

ax[3].imshow(img_out)
ax[3].set_title('Output')

plt.tight_layout()
plt.savefig('SeamCarving_Full.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
plt.show()

#################################################################

imname = input('Enter Image Name (with directory) : ')
img = cv2.imread(imname, 0)
print('The image has a resolution of (X Y) : ', img.shape[0], img.shape[1])
nshape = input('Enter New Resolution for Image (X and Y seperated by space) : ')
nshape = nshape.split()
nshape = np.array(nshape, dtype='int32')

# Driver Function
bkp_e, bkp_vc, bkp_hc, img_out = sc_resize_image(img, nshape)

write_energy('in_img_energy.txt', bkp_e)

plt.imshow(img, cmap='gray')
plt.savefig('00_Image.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
plt.show()

plt.imshow(bkp_e)
plt.savefig('01_Energy.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
plt.show()

plt.imshow(bkp_vc)
plt.savefig('02_VEnergy.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
plt.show()

plt.imshow(bkp_hc)
plt.savefig('03_HEnergy.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
plt.show()

plt.imshow(img_out, cmap='gray')
plt.savefig('04_Output.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
plt.show()

fig, ax = plt.subplots(1, 5, figsize=(15, 15))

ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original')

ax[1].imshow(bkp_e)
ax[1].set_title('Energy')

ax[2].imshow(bkp_vc)
ax[2].set_title('VEnergy')

ax[3].imshow(bkp_hc)
ax[3].set_title('HEnergy')

ax[4].imshow(img_out, cmap='gray')
ax[4].set_title('Output')

plt.tight_layout()
plt.savefig('SeamCarving_Full.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
plt.show()
