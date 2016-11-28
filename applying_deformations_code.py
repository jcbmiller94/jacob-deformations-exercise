""" Applying deformations exercise
"""

#: standard imports
import numpy as np
import matplotlib.pyplot as plt
# print arrays to 4 decimal places
np.set_printoptions(precision=4, suppress=True)
import numpy.linalg as npl
import nibabel as nib

#: gray colormap and nearest neighbor interpolation by default
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'nearest'

#: load y_ds107_sub012_highres.nii with nibabel
# get the image array data
deformations_img = nib.load('y_ds107_sub012_highres.nii')
deformations_data = deformations_img.get_data()
#deformations_data.shape

#: remove length-1 4th dimension from deformation data
deformations_data = np.squeeze(deformations_data)
#deformations_data.shape

#: get original TPM.nii 3D shape and affine
tpm_shape = deformations_data.shape[:3]
tpm_affine = deformations_img.affine
#tpm_affine

#: load the template image we will resample from
template_img = nib.load('mni_icbm152_t1_tal_nlin_asym_09a.nii')
template_data = template_img.get_data()
#template_data.shape

#: voxels in TPM.nii to voxels in mni_icbm152_t1_tal_nlin_asym_09a.nii
# Matrix multiplication is right to left
vox2vox = npl.inv(template_img.affine).dot(tpm_affine)
#vox2vox

#: to mat and vec
mat, vec = nib.affines.to_matvec(vox2vox)
#mat
#vec

#: resample MNI template onto TPM grid
from scipy.ndimage import affine_transform
template_into_tpm = affine_transform(template_data, mat, vec,
                                     output_shape=tpm_shape)
#template_into_tpm.shape

#: plot the template image resampled onto the TPM grid
plt.imshow(template_into_tpm[:, :, 60])

#: load the subject data that we will resample from
subject_img = nib.load('ds107_sub012_highres.nii')
subject_data = subject_img.get_data()
#subject_data.shape

#- * get mapping from voxels in TPM to voxels in the subject image;
#- * resample the subject image into the grid of the TPM image using
#-   this mapping.

# tpm2subj = npl.inv(subject_img.affine).dot(deformations_data)
# ^ tried to do the above first, but because the shapes do not align I had to
#  use affine transform, as with the solutions
vox2vox_mapping = nib.affines.apply_affine(npl.inv(subject_img.affine), deformations_data)
#I, J, K = subject_data.shape
#i_vals, j_vals, k_vals = np.meshgrid(range(I), range(J), range(K), indexing='ij')
#in_vox_coords = np.array([i_vals, j_vals, k_vals])
#coords_last = in_vox_coords.transpose(1, 2, 3, 0)
# ^ thought I needed all this, turns out meshgrid not applicable here
#   because deformations_data already had some form of mapping done through SPM.
# This is what took me so long to wrap my head aorund here, the fact that we are
#  working with a deformation and not a true image as the example showed
transposed_coords = vox2vox_mapping.transpose(3, 0, 1, 2)
# Resample using map_coordinates
from scipy.ndimage import map_coordinates
resampled_subj = map_coordinates(subject_data, transposed_coords)
# Resampling the subject T1 into the space of the TPM, by passing it the
#   coordinate grid of the voxel to voxel mapping of the sbuject affine and the deformations
#   needed to turn the subject into template space (from SPM)
# My biggest question is - ow would we approach this without the deformation map?
#  I suppose it would be more similar to the example exercise, just with resampling
#  the structural subject image into a template space, and using meshgrid (we would need
#  the info SPM has about the template... )

# Show resampled data
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(template_into_tpm[:, :, 50])
axes[1].imshow(resampled_subj[:, :, 50])
plt.show()
#I, J, K = tpm_shape
#i_vals, j_vals, k_vals = np.meshgrid(range(I), range(J), range(K),
#                                     indexing='ij')
#- show an example slice from the resampled template and resampled
#- subject
