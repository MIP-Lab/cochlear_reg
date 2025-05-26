# cochlear_reg

This repo implements a deep-learning-based registration network for segmenting cochlear structures (modiolus, scala tympani, scala vestibuli, the labyrinth) in metal-artifact-affected CT images (post-cochlear-implantation images).

### 1. Requirements
System: Windows or Linux\
Python >= 3.8\
pytorch >= 1.12\
pytorch3d >= 0.7.0 (for computing the chamfer loss)\
pyvista >= 0.43.0, matplotlib >= 3.7.5 (for visualizing the results)\
jupyter_core >= 5.7.2 (for reproducing the statistical results)

### 2. Inference with the proposed chamfer model.

Step1: Download the model checkpoints and them inside the checkpoints folder.

Step2: Prepare the image crops around the ear in the nifti format and put them in a folder. The provided model uses 128x128x128 input volumes with 0.2x0.2x0.2 mm resolution. An example of the image crop is shown below. The model is trained for post-cochlear-implantation images but is supposed to work reasonably well for pre-cochlear-implantation images, i.e., images without metal artifacts.

Step3: Run 
```
python inference_chamfer_model --data_folder <absolute_url_to_the_test_nifti_files>
```

### 3. Training the proposed chamfer model.

Step1: Prepare the image crops around the ear in the nifti format and put them in separate training and validation folders. 128x128x128 input volumes with 0.2x0.2x0.2 mm resolution were used the our work. But other sizes and resolutions can also be used.

Step2: Run 
```
python train_chamfer_model \
--train_folder <absolute_url_to_the_training_nifti_files> \
--val_folder <absolute_url_to_the_validation_nifti_files> \
--experiment_name <identifier_to_save_the_checkpoints_and_training_losses>
```

### 4. Reproducing the main statstical results reported in the paper.

Step1: Download the predicted meshes and segmentation masks obtained with different methods: the chamfer, p2p, dice models, and cGAN-ASM, nnU-Net, SegNet, ABA and Elastix methods from https://doi.org/10.5281/zenodo.15519545

Step2: Run reproducibility/figures.ipynb.

### 5. Reproducing the predicted meshes and binary masks.

Ideally, we would like to be able to reproduce the predicted meshes and segmentation masks used in section 4 from the original test CT images used in the paper. However, the images we used were from a private dataset that cannot be publicly shared. Here we provide a compromised way to provide the reproducbility in terms of meshes and segmentation masks: we obtain and upload the activation maps through the first convolution layers of different models, and someone can download the 
activation maps and pass them through the subsequent network layers to produce the predicted meshes and segmentation masks. This approach works for the chamfer, p2p, dice, nnU-Net, SegNet models. It does not work for cGAN-ASM, ABA or Elastix because they require working on the original images and because some of the codes with these methods cannot be public shared either.

Step1: Download the activation maps from the following links:

Step2: Run the following to obtain the predicted meshes with the chamfer, p2p, and dice models, and the predicted segmentation masks with nnU-Net and SegNet.
```
python reproducibility generate_prediction_from_activation_maps.py
```
Step3: Use ```reproducibility/generate_binary_masks_from_meshes.py``` to convert the predicted meshe obtained with the chamfer, p2p, and dice models to binary masks for the evaluation of DICE (this step needs to be in Windows system becauses it relies on a .exe file).
```
python reproducibility generate_binary_masks_from_meshes.py
```
Step4: Use ```reproducibility/generate_meshes_from_binary_masks.py``` to convert the predicted segmentation masks obtained with the nnU-Net and Segment to meshes. Then use ```reproducibility/atlas_to_patient_mesh_registration.m``` to register the atlas meshes to the mask-converted patient meshes for the evaluation of the point-to-point correspondence error.

The results generated from the above steps shoud be the same as those used in section 4.

