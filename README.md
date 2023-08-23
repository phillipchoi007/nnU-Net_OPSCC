#nnU-Net_OPSCC
Brief code for training multi-modal deep learning segmentation models of oropharyngeal squamous cell carcinoma on CT and MRI using nnU-Net framework. 
Details of the background and running inference is here (https://github.com/MIC-DKFZ/nnUNet). 

#The overall workflow of multi-modal segmentation model
![FIG2](https://github.com/phillipchoi007/nnU-Net_OPSCC/assets/40045450/dcb539dd-1f04-41c5-89d8-49b55338b041)

#activate virtual environment
cd ~/Documents 
source nnunet22/bin/activate

#set environment variables (run everytime prior to each training)
export nnUNet_raw_data_base="/mnt/ysnas/supraclavicular_CT_done/nnunet/nnunet/nnUNet_raw_data_base"
export nnUNet_preprocessed="/mnt/ysnas/supraclavicular_CT_done/nnunet/nnunet/preprocessed"
export RESULTS_FOLDER="/mnt/ysnas/supraclavicular_CT_done/nnunet/nnunet/nnUNet_trained_models"

#Task201: CT model
#Task301: MR model
#Task401: CT-MR model

#TIPS: for multi-modal training, make sure the task number changes in hundreds only (ex: Task101, Task201, Task301, and not Task201, Task202, etc!)

#check data integrity
nnUNet_plan_and_preprocess -t 201 --verify_dataset_integrity #CT
nnUNet_plan_and_preprocess -t 301 --verify_dataset_integrity #MR
nnUNet_plan_and_preprocess -t 401 --verify_dataset_integrity #MR&CT

#train using four GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3

##3d_fullres model training (be aware of the GPU memory as each line of code takes up around 4-8 GB)
#CT (five cross-validation)

CUDA_VISIBLE_DEVICES=1 nnUNet_train 3d_fullres nnUNetTrainerV2 Task201_tonsil 0 --npz
CUDA_VISIBLE_DEVICES=1 nnUNet_train 3d_fullres nnUNetTrainerV2 Task201_tonsil 1 --npz
CUDA_VISIBLE_DEVICES=1 nnUNet_train 3d_fullres nnUNetTrainerV2 Task201_tonsil 2 --npz
CUDA_VISIBLE_DEVICES=1 nnUNet_train 3d_fullres nnUNetTrainerV2 Task201_tonsil 3 --npz
CUDA_VISIBLE_DEVICES=2 nnUNet_train 3d_fullres nnUNetTrainerV2 Task201_tonsil 4 --npz

#MR (five cross-validation)

CUDA_VISIBLE_DEVICES=1 nnUNet_train 3d_fullres nnUNetTrainerV2 Task301_tonsil 0 --npz
CUDA_VISIBLE_DEVICES=1 nnUNet_train 3d_fullres nnUNetTrainerV2 Task301_tonsil 1 --npz
CUDA_VISIBLE_DEVICES=1 nnUNet_train 3d_fullres nnUNetTrainerV2 Task301_tonsil 2 --npz
CUDA_VISIBLE_DEVICES=2 nnUNet_train 3d_fullres nnUNetTrainerV2 Task301_tonsil 3 --npz
CUDA_VISIBLE_DEVICES=2 nnUNet_train 3d_fullres nnUNetTrainerV2 Task301_tonsil 4 --npz

#MR/CT (five cross-validation)

CUDA_VISIBLE_DEVICES=0 nnUNet_train 3d_fullres nnUNetTrainerV2 Task401_tonsil 0 --npz
CUDA_VISIBLE_DEVICES=0 nnUNet_train 3d_fullres nnUNetTrainerV2 Task401_tonsil 1 --npz
CUDA_VISIBLE_DEVICES=0 nnUNet_train 3d_fullres nnUNetTrainerV2 Task401_tonsil 2 --npz
CUDA_VISIBLE_DEVICES=3 nnUNet_train 3d_fullres nnUNetTrainerV2 Task401_tonsil 3 --npz
CUDA_VISIBLE_DEVICES=2 nnUNet_train 3d_fullres nnUNetTrainerV2 Task401_tonsil 4 --npz

##2d model training 
#CT (five cross-validation)

CUDA_VISIBLE_DEVICES=2 nnUNet_train 2d nnUNetTrainerV2 Task201_tonsil 0 --npz
CUDA_VISIBLE_DEVICES=2 nnUNet_train 2d nnUNetTrainerV2 Task201_tonsil 1 --npz
CUDA_VISIBLE_DEVICES=3 nnUNet_train 2d nnUNetTrainerV2 Task201_tonsil 2 --npz
CUDA_VISIBLE_DEVICES=3 nnUNet_train 2d nnUNetTrainerV2 Task201_tonsil 3 --npz
CUDA_VISIBLE_DEVICES=0 nnUNet_train 2d nnUNetTrainerV2 Task201_tonsil 4 --npz

#MRI (five cross-validation)

CUDA_VISIBLE_DEVICES=2 nnUNet_train 2d nnUNetTrainerV2 Task301_tonsil 0 --npz
CUDA_VISIBLE_DEVICES=3 nnUNet_train 2d nnUNetTrainerV2 Task301_tonsil 1 --npz
CUDA_VISIBLE_DEVICES=3 nnUNet_train 2d nnUNetTrainerV2 Task301_tonsil 2 --npz
CUDA_VISIBLE_DEVICES=3 nnUNet_train 2d nnUNetTrainerV2 Task301_tonsil 3 --npz
CUDA_VISIBLE_DEVICES=0 nnUNet_train 2d nnUNetTrainerV2 Task301_tonsil 4 --npz

#MRI/CT (five cross-validation)

CUDA_VISIBLE_DEVICES=1 nnUNet_train 2d nnUNetTrainerV2 Task401_tonsil 0 --npz
CUDA_VISIBLE_DEVICES=1 nnUNet_train 2d nnUNetTrainerV2 Task401_tonsil 1 --npz
CUDA_VISIBLE_DEVICES=1 nnUNet_train 2d nnUNetTrainerV2 Task401_tonsil 2 --npz
CUDA_VISIBLE_DEVICES=3 nnUNet_train 2d nnUNetTrainerV2 Task401_tonsil 3 --npz
CUDA_VISIBLE_DEVICES=2 nnUNet_train 2d nnUNetTrainerV2 Task401_tonsil 4 --npz

#if training process is interrupted:
nnU-Net stores a checkpoint every 50 epochs. If you need to continue a previous training, just add a -c to the training command.
nnUNet_train CONFIGURATION TRAINER_CLASS_NAME TASK_NAME_OR_ID FOLD  --npz -c

#Find the best U-Net configuration
nnUNet_find_best_configuration -m 2d 3d_fullres 3d_lowres 3d_cascade_fullres -t XXX

#In this case
nnUNet_find_best_configuration -m 2d 3d_fullres -t 201
nnUNet_find_best_configuration -m 2d 3d_fullres -t 301
nnUNet_find_best_configuration -m 2d 3d_fullres -t 401
