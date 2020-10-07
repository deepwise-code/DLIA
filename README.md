## DLIA
create time: 2020.08.26

#### Introduction
This repository contains the source code and the trained model for our work Deep Learning for Intracranial Aneurysm Detection in Computed Tomography Angiography Images.

Prerequisites
* Ubuntu: 16.04 lts
* Python 3.6.5
* Pytorch 1.0.1.post2
* NVIDIA GPU + CUDA_10.0 CuDNN_7.5



This repository has been tested on NVIDIA RTX2080Ti. Configurations (e.g batch size, image patch size) may need to be changed on different platforms.

#### Installation

Other packages are as follows:
* yacs
* nibabel
* scipy
* joblib
* opencv-python
* SimpleITK
* scikit-image
* numpy

Install dependencies:
```shell script
pip install -r requirements.txt
```

#### Usage
We mainly use main.py (under the project) to train/val/test our models.

The following is one example:
```shell script
python main.py --gpu 0 1 2 3  --train --config tasks/configs/aneurysm_seg.daresunet.yaml
```
The main parameters are following:

* --train: used to train the model.
* --test: used to test(val) the model.
* --config: the path to the configuration file(*.yaml).
* --resume(optional): the path to the checkpoint pth(resume the model).
* --gpu(default 0): decide to which gpu to select. Format: one or multiple integers(separated by space keys), such as
* --gpu 0 1 2 3
* --check_point(optional): the path to save the trained model, we usually specify the parameter in the config file, if you specify this parameter here, it will override this parameter in the config file.

###### Train
Run command as below.
```shell script
python main.py --gpu 0 1 2 3  --train --config tasks/configs/aneurysm_seg.daresunet.yaml
```

##### Inference
the weight of the model: `raws/weight/da_resunet.pth.tar`

configuration file: `tasks/configs/aneurysm_seg.daresunet.yaml`
* `TEST.DATA.NII_FOLDER`: directory of input files
* `TEST.DATA.TEST_FILE`: list of file names
* `TEST.SAVE_DIR`: the directory to save results
If you want to use your custom data, you need modify the yaml file to set the path and file names of the test data. 

Run command as below.
```shell script
python main.py --gpu 0 1 2 3  --test --config tasks/configs/aneurysm_seg.daresunet.yaml --check_point raws/weight/da_resunet.pth.tar
```
##### qucik start

you can quick start inference and visualization using `vis_demo.ipynb`

* the example image data: `raws/image/example.nii.gz`
* the ground truth : `raws/image/example_mask.nii.gz`
* the expected output:  `raws/example_seg.nii.gz`

#### other

in the `./raws` directory, we supply an example dataset
```shell script
raws
├── image  # directory of image files
├── mask   # directory of ground truth
├── lesion_bbox.txt  # lesion bbox info 
├── part_train.txt  # train list
├── part_val.txt    # validation list
├── part_test.txt   # test list 
```