
# cochlear_reg

This repository implements a deep learning–based method for segmenting cochlear structures—**modiolus**, **scala tympani**, **scala vestibuli**, and the **labyrinth**—in CT images affected by metal artifacts (e.g., post-cochlear-implantation scans). The method works by doing image registration between atlas and patient images, and then deforming the pre-defined atlas meshes into the patient space.

![image](https://github.com/user-attachments/assets/3f15ba4a-7c80-4fa0-b255-b660d9ef8fed)  
![image](https://github.com/user-attachments/assets/2c474172-63ea-4599-92dd-08c142564f85)

---

## 1. Requirements

- **OS:** Windows or Linux  
- **Python:** ≥ 3.8  
- **PyTorch:** ≥ 1.12  
- **PyTorch3D:** ≥ 0.7.0 *(used for Chamfer loss)*  
- **PyVista:** ≥ 0.43.0 and **Matplotlib:** ≥ 3.7.5 *(for visualization)*  
- **jupyter_core:** ≥ 5.7.2 *(for reproducing statistical analysis)*

---

## 2. Inference Using the Proposed Chamfer Model

### Step 1
Download the pretrained model checkpoints from [Zenodo](https://doi.org/10.5281/zenodo.15520931) and place them in the `checkpoints/` folder.

### Step 2
Prepare 3D image crops of the ear in **NIfTI format** (128×128×128, resolution: 0.2×0.2×0.2 mm) and place them in a folder.  
The model is trained on post-implantation scans but performs reasonably well on pre-implantation (metal-free) images.

Example of a sagittal slice of the cochlear image crop (red = binary mask of the labyrinth):

<img src="https://github.com/user-attachments/assets/01855ae1-c85d-4a79-b4ce-7520f4372a00" width="400">

### Step 3
Run the inference command:
```bash
python inference_chamfer_model --data_folder <absolute_path_to_test_nifti_files>
```

---

## 3. Training the Proposed Chamfer Model on Custom Datasets

### Step 1
Organize training and validation image crops (in NIfTI format) into separate folders, for example, ```training/image``` and ```validation/image```.  
While 128×128×128 volumes at 0.2 mm resolution were used in our work, other configurations are supported by the network structure.

### Step 2
Prepare the meshes of the anatomical structures for both training and validation samples and put them into the corresponding folder, for example, ```training/mesh/structureA``` and ```validation/mesh/structureA```. The meshes can be in the common .obj file format or our custom .mesh file format, of which the writing and read code is provided in ```dataset/cochlear_ct.py```. The meshes can be converted from binary segmentation masks using the marching cube algorithm. An example code is provided in ```reproducibility/generate_meshes_from_binary_masks.py```
While 128×128×128 volumes at 0.2 mm resolution were used in our work, other configurations are supported by the network structure.

### Step 3
Run training:
```bash
python train_chamfer_model \
--structure_names (e.g., structureA|structureB|structureC, different structures separated by '|', and names consistent with the convention used in Step 2)
--train_folder <absolute_path_to_training_data> \
--val_folder <absolute_path_to_validation_data> \
--experiment_name <experiment_id_for_checkpoints_and_logs>
```

---

## 4. Reproducing Main Statistical Results

### Step 1
Download predicted meshes and segmentation masks from [Zenodo](https://doi.org/10.5281/zenodo.15519545).  
These include outputs from Chamfer, P2P, Dice, cGAN-ASM, nnU-Net, SegNet, ABA, and Elastix.

### Step 2
Run the following Jupyter notebook:
```
reproducibility/figures.ipynb
```

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

---

### Step 2: Generate Predictions

```bash
python reproducibility/generate_prediction_from_activation_maps.py
```

---

### Step 3: Convert Meshes to Binary Masks (Windows only)

Convert predicted meshes (from Chamfer, P2P, Dice) into binary segmentation masks for DICE evaluation:
```bash
python reproducibility/generate_binary_masks_from_meshes.py
```

---

### Step 4: Convert Masks to Meshes

Convert masks (from nnU-Net, SegNet) to meshes:
```bash
python reproducibility/generate_meshes_from_binary_masks.py
```

Then register the atlas mesh to patient-specific meshes in MATLAB:
```matlab
reproducibility/atlas_to_patient_mesh_registration.m
```

---

**Note:** The results generated from the above steps should match those downloaded in section 4 of this document.
