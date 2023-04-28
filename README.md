# MSc_FR_Bias

This repository contains the source code for the implementation of the master thesis. The master thesis investigates the use of synthetic data to fine-tune a face recognition system in order to attempt to reduce the system's bias towards age. 

### Table of Contents
- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Inital Setup](#inital-setup)
  - [Creating a virtual enviorment](#creating-a-virtual-enviorment)
  - [Installing the required packages](#installing-the-required-packages)
- [Running the code](#running-the-code)
    - [Creating synthetic face images of different ages using CUSP](#creating-synthetic-face-images-of-different-ages-using-cusp)
    - [Creating synthetic face images of different ages using EG3D-Age](#creating-synthetic-face-images-of-different-ages-using-eg3d-age)
    - [Fine-tuning the ArcFace model on a synthetic dataset](#fine-tuning-the-arcface-model-on-a-synthetic-dataset)

### Features

* Create synthetic face images using the CUSP repository
* Create synthetic face images using the EG3D-Age repository
* Fine-tune ArcFace on synthetic face images in an attempt to mitigate age bias and increase the fairness of the face recogntion system
* Evaluate the performance of the fine-tuned model thorugh distribution plots and the GARBE metric.

### Installation

#### Prerequisites
In order to run this repository, one needs the following tools:

* pyhton3
* pip 

#### Inital Setup
First, clone the GitHub repository for the MSc_FR_Bias project using the following command:

```
git clone https://github.com/pernilko/MSc_FR_Bias.git
```

Once the project has been cloned, create a folder within the `src`-folder of the project with the name models. This can be done with the following command:

```
mkdir models
```

Inside this folder, a series of other repositories needs to be downloaded:
* InsightFace: `https://github.com/deepinsight/insightface`
* EG3D-Age: `https://github.com/johndoe133/eg3d-age`
* CUSP: `https://github.com/guillermogotre/CUSP`

Inside the models-folder, the InsightFace repository should be download to a folder named `insightface2`, the EG3D-Age repository should be downloaded to a folder named `eg3dAge`, and the CUSP repository should be downloaded to a folder named `cusp`.

Once these repositories have been downloaded and their folders are appropriatly named, the pre-trained ArcFace model from the InsightFace repository needs to be downloaded into the `models`-folder. The ArcFace model can be found in the model zoo of the `arcface_torch` InsightFace repository (`https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch`). In the README file contained in this project, there is a model zoo section which provides a link to a OneDrive. In the OneDrive, select the folder named `ms1mv3_arcface_r50_fp16`. Inside this folder, there is a file named `backbone.pth`. Download this file and store it in the `models`-folder that was created inside the `src`-folder of the `MSc_FR_Bias`-project.


Finally, create a new folder in the `src`-folder of the project. This folder should be named `datasets`. This can be done through the following command:
```
mkdir datasets
```

Once the folder has been created, the FG-NET dataset needs to be downloaded and stored to a folder called `fgnet`. The FG-NET dataset can be acquired from `INSERT LINK`. 

#### Creating a virtual enviorment

First, check if virtualenv is already installed using the following command:

```
which virtualenv
```

If virtualenv is not already installed, install it using the following command:

```
pip install virtualenv
```

Next, create a virtual enviroment in the src folder of the cloned MSc_FR_Bias project. To do this, run the command 

```
virtualenv <env_name>
```

Finally, it is time to activate the virtual enviorment. Use the following command to activate the virutal enviorment:

```
source <env_name>/bin/activate
```

#### Installing the required packages
Once the virtual enviorment has been created, run the following command to install the required packages:

```
pip install -r requirements.txt
```

### Running the code

#### Creating synthetic face images of different ages using CUSP


#### Creating synthetic face images of different ages using EG3D-Age

#### Fine-tuning the ArcFace model on a synthetic dataset

Navigate into the `src`-folder of the `MSc_FR_Bias`-project. Then, run the following command

```
python3 -m optimization.fine_tuning_pipeline --input_img_path=datasets/eg3d_generated/ --dist_plot_path=plots/ --lr=0.01 --momentum=0.9 --epochs=150 --batch_size=25 --frozen_layers=1,2,3 --experiment_name=eg3d_exp_1 --test_data_path=datasets/fgnet/
```



