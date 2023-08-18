# Manas
This repository contains the source code, benchmark models, and datasets for the paper - **"Manas: Mining Software Repositories to Assist AutoML"**, appeared in ICSE 2022 at Pittsburgh, Pennsylvania, United States.

### Authors
* Giang Nguyen, Iowa State University (gnguyen@iastate.edu)
* Md Johirul Islam, Amazon (johirbuet@gmail.com)
* Rangeet Pan, IBM Research (rangeet.pan@ibm.com)
* Hridesh Rajan, Iowa State University (hridesh@iastate.edu)
  
### Abstract
Today deep learning is widely used for building software. A software engineering problem with deep learning is that finding an appropriate convolutional neural network (CNN) model for the task can be a challenge for developers. Recent work on AutoML, more precisely neural architecture search (NAS), embodied by tools like Auto-Keras aims to solve this problem by essentially viewing it as a search problem where the starting point is a default CNN model, and mutation of this CNN model allows exploration of the space of CNN models to find a CNN model that will work best for the problem. These works have had significant success in producing high-accuracy CNN models. There are two problems, however. First, NAS can be very costly, often taking several hours to complete. Second, CNN models produced by NAS can be very complex that makes it harder to understand them and costlier to train them. We propose a novel approach for NAS, where instead of starting from a default CNN model, the initial model is selected from a repository of models extracted from GitHub. The intuition being that developers solving a similar problem may have developed a better starting point compared to the default model. We also analyze common layer patterns of CNN models in the wild to understand changes that the developers make to improve their models. Our approach uses commonly occurring changes as mutation operators
in NAS. We have extended Auto-Keras to implement our approach. Our evaluation using 8 top voted problems from Kaggle for tasks including image classification and image regression shows that given the same search time, without loss of accuracy, Manas produces models with 42.9% to 99.6% fewer number of parameters than Auto-Keras’ models. Benchmarked on GPU, Manas’ models train 30.3% to 641.6% faster than Auto-Keras’ models.

![The problem tackled by Manas](overview.JPG)
### Environment Setup
To run Fair-AutoML, we need to install Python 3 environment on Linux. 

### Environment Setup
Follow these steps to clone the Fair-AutoML repository and install Fair-AutoML.

1. Clone this repository and move to the directory:

```
git clone https://github.com/giangnm58/Manas.git
cd Manas/
``` 

2. Navigate to the cloned repository: `cd Manas/` and install required packages:

```
pip install -r requirements.txt
```

To run the tool, please refer to the [installation file](/INSTALL.md) for detailed instructions. 



### Cite the paper as
```
@inproceedings{nguyen2022manas,
  title={Manas: Mining software repositories to assist automl},
  author={Nguyen, Giang and Islam, Md Johirul and Pan, Rangeet and Rajan, Hridesh},
  booktitle={Proceedings of the 44th International Conference on Software Engineering},
  pages={1368--1380},
  year={2022}
}
```
