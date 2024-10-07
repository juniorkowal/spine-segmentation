## imports

# libraries
import os
import numpy as np
import nibabel as nib
import nibabel.orientations as nio
import matplotlib.pyplot as plt

# custom
import data_utilities as dutils
import eval_utilities as eutils 


## paths

directory = os.path.join(os.getcwd(),'sample')

true_msk = nib.load(os.path.join(directory,'sub-verse004_seg-vert_msk.nii.gz')) 
pred_msk = nib.load(os.path.join(directory,'sub-verse004_seg-vert_msk.nii.gz')) # use the same file for example

true_ctd = dutils.load_centroids(os.path.join(directory,'sub-verse004_seg-subreg_ctd.json'))
pred_ctd = dutils.load_centroids(os.path.join(directory,'sub-verse004_seg-subreg_ctd.json'))


## pre-process (evaluation was done at 1mm because annotations were performed at 1mm)

true_msk = dutils.resample_nib(true_msk, voxel_spacing=(1, 1, 1), order=0) # or resample based on img: resample_mask_to(msk_nib, img_iso)
true_ctd = dutils.rescale_centroids(true_ctd, true_msk, (1,1,1))

pred_msk = dutils.resample_nib(pred_msk, voxel_spacing=(1, 1, 1), order=0) # or resample based on img: resample_mask_to(msk_nib, img_iso)
pred_ctd = dutils.rescale_centroids(pred_ctd, pred_msk, (1,1,1))



## compute dice

true_msk_arr =  true_msk.get_fdata()
pred_msk_arr =  pred_msk.get_fdata()


dice = eutils.compute_dice(pred_msk_arr, true_msk_arr)

print('Dice:{:.2f}'.format(dice))


## compute id_rate

MAX_VERT_IDX = 28 # (in VerSe20, T13 has an index of 28)

# create an array of shape (MAX_VERT_IDX, 3), 
# i_th row contain centroid for (i+1)_th vertebra. Rest are NaNs

def prepare_ctd_array(ctd_list, max_vert_idx):
    ctd_arr = np.full((max_vert_idx, 3), np.nan)
    for item in ctd_list[1:]: # first entry contains orientation 
        vert_idx = item[0]
        if vert_idx <= max_vert_idx:
            X = item[1]
            Y = item[2]
            Z = item[3]
            ctd_arr[vert_idx - 1, :] = [X, Y, Z]
    return ctd_arr

true_ctd_arr =  prepare_ctd_array(true_ctd, MAX_VERT_IDX)
pred_ctd_arr =  prepare_ctd_array(pred_ctd, MAX_VERT_IDX)

# get number of successful hits (identifications)

num_hits, hit_list = eutils.get_hits(true_ctd_arr, pred_ctd_arr, MAX_VERT_IDX)
verts_in_gt        = np.argwhere(~np.isnan(true_ctd_arr[:, 0])).reshape(-1) + 1  # list of vertebrae present in annotation

print('id.rate:{:.2f}\n'.format(num_hits/len(verts_in_gt)))
print('Hits:\n', hit_list) # nan : vertebrae is absent. 1 : successful identifcation. 0 : failed identification