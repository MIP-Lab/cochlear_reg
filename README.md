
# cochlear_reg

This repository implements a deep learning–based method for segmenting cochlear structures—**modiolus**, **scala tympani**, **scala vestibuli**, and the **labyrinth**—in CT images affected by metal artifacts (e.g., post-cochlear-implantation scans). The method works by doing image registration between atlas and patient images, and then deforming the pre-defined atlas meshes into the patient space.

![image](https://github.com/user-attachments/assets/3f15ba4a-7c80-4fa0-b255-b660d9ef8fed)  
![image](https://github.com/user-attachments/assets/2c474172-63ea-4599-92dd-08c142564f85)

---

## 1. Requirements

- **OS:** Windows or Linux  
- **Python:** ≥ 3.8
- **Nibabel:** ≥ 4.0.1
- **PyTorch:** ≥ 1.12  
- **PyTorch3D:** ≥ 0.7.0 *(used for Chamfer loss)*  
- **PyVista:** ≥ 0.43.0 and **Matplotlib:** ≥ 3.7.5 *(for visualization)*  
- **jupyter_core:** ≥ 5.7.2 *(for reproducing statistical analysis)*

---

## 2. Inference Using the Proposed Chamfer Model

### Step 1
Download the pretrained model checkpoints and the atlas data (including activation map from the chamfer model and mesh points, but not the raw atlas image) we used from [Zenodo](https://doi.org/10.5281/zenodo.15520931). Put the checkpoints folders under ./checkpoints and the atlas folder under ./data.

### Step 2
Prepare 3D image crops of the ear in **NIfTI format** (128×128×128, resolution: 0.2×0.2×0.2 mm, intensity rescaled between -1 and 1) and place them in a folder.  
The model is trained on post-implantation scans but performs reasonably well on pre-implantation (metal-free) images.

Example of a sagittal slice of the cochlear image crop (red = binary mask of the labyrinth):

<img src="https://github.com/user-attachments/assets/01855ae1-c85d-4a79-b4ce-7520f4372a00" width="400">

### Step 3
Run the inference command:
```bash
python inference_chamfer_model_activation.py --nii_folder <absolute_path_to_test_nifti_files>
```
---

## 3. Training the Proposed Chamfer Model on Custom Datasets

### Step 1
Prepare the training images crops in NIfTI format. While 128×128×128 volumes at 0.2 mm resolution were used in our work, other configurations are supported by the network structure.

### Step 2
Prepare the meshes of the anatomical structures for each training image, then concate the vertices from all structure meshes and save them into a .pkl file named the same as its corresponding image (e.g., patient01.nii.gz and patient01.pkl). 

Suppose you have a binary mask in NifTI format of a certain structure, you can prepare the .pkl file as follows,

```
import numpy as np
import pickle
import nibabel as nib
from skimage import measure

# convert the binary mask into meshes
# If you have multiple masks (for different structures), convert them to verts one by one and concate all the vertices into a large Nx3 numpy array
mask_data = np.array(nib.load('mask.nii.gz'))
verts, faces, _, _ = measure.marching_cubes(mask_data, 0)

# save the vertices in a .pkl file
pickle.dump(nx3_numpy_array, open(vertices.pkl, 'wb'))

# load from .pkl file
np.load(vertices.pkl, 'rb', allow_pickle=True)
```

### Step 3
Run training:
```bash
python train_chamfer_model.py \
--nii_folder <path_to_nifti_files> \
--vtx_folder <path_to_vtx_pkl_files> \
--atlas_name <name_of_the_atlas_case> \
--experiment_name <experiment_id_for_checkpoints_and_logs>
```
Note that 20% of the training cases will be used as the validation set automatically.

---

## 4. Reproducing Main Statistical Results

### Step 1
Download the predicted meshes and segmentation masks from [Zenodo](https://doi.org/10.5281/zenodo.15519545).  

### Step 2
Run ```reproducibility/plot_meshes_different_methods.py``` to reproduce the visual results reported in Fig. 6 of the paper.\
Run the ```reproducibility/figures.ipynb``` Jupyter notebook to reproduce the quantitative results reported in Fig. 4 & 5 of the paper.

Need to run them inside the ```reproducibility``` folder.

---

## 5. Reproducing Predicted Meshes and Masks

The original test CT images cannot be shared publicly due to dataset restrictions.  
Instead, we provide **activation maps** from the first convolutional layers of the models. These can be used to reproduce outputs used in section 4 by forwarding them through the remaining layers.

This approach supports:
- Chamfer
- P2P
- Dice
- nnU-Net
- SegNet

It does **not** support:
- cGAN-ASM
- ABA
- Elastix  
(due to reliance on original images or closed-source code)

---

### Step 1: Download Activation Maps

Place them under `data/activation_maps/`:

- [Chamfer](https://doi.org/10.5281/zenodo.15519630)  
- [P2P](https://doi.org/10.5281/zenodo.15520101)  
- [Dice](https://doi.org/10.5281/zenodo.15519921)  
- [SegNet](https://doi.org/10.5281/zenodo.15520369)
- [nnU-Net-MD](https://doi.org/10.5281/zenodo.15531266), [nnU-Net-STSV](https://doi.org/10.5281/zenodo.15531309), [nnU-Net-Labyrinth](https://doi.org/10.5281/zenodo.15531303)
---

### Step 2: Generate Predictions

```bash
python reproducibility/generate_prediction_from_activation_maps.py
```
For nnU-Net, the process is different: \
(1) Install the nn-UNet v1 from their github repository https://github.com/MIC-DKFZ/nnUNet. Checkout the v1 branch rather than the master branch for installation. \
(2) Set environment variables required by nnunet according to https://github.com/MIC-DKFZ/nnUNet/blob/nnunetv1/documentation/setting_up_paths.md. The ```RESULTS_FOLDER``` to needs to point to the ```checkpoints```folder of this repo. \
(3) Replace the original ```inference/predict.py``` in the nnunet code under the ```site-packages``` with our ```reproducibility/nnunet_predict.py``` (keep the original file name). This will allow the nnunet to use activation maps rather than the original images. \
(4) Run the standard nnunet prediction command: \

```bash
nnUNet_predict \
-chk model_best \
-tr nnUNetTrainerV2 \
-i <path_to_the_downloaded_nnunet_activation_maps> \
-o <path_to_save_the_predicted_binary_masks> \
-t <602 or 604 or 606 > \ # (602 for MD, 604 for STSV and 606 for the labyrinth) \
-m 3d_fullres \
-f 0 \
```

---

### Step 3: Convert between Mesh and Binary Masks (Windows only).
```
python reproducibility/post_processing_acvitation_maps.py
```

### Step 4: Do mesh registration between atlas and the mask-converted patient meshes for P2P error evaluation (Matlab).

---
