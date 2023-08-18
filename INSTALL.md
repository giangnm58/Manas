
## Installation
### Environment Setup
To run Fair-AutoML, we need to install Python 3 environment on Linux. Follow these steps to clone the Fair-AutoML repository and install Fair-AutoML.

1. Clone this repository and move to the directory:

```
git clone https://github.com/giangnm58/Manas.git
cd Manas/
``` 

2. Navigate to the cloned repository: `cd Manas/` and install required packages:

```
pip install -r requirements.txt
```
## Run the Manas tool

For optimal program execution, we employ PyCharm, a user-friendly integrated development environment.

To acquire PyCharm, visit https://www.jetbrains.com/pycharm/ and proceed with the installation.

Here's a concise guide to getting started:

Step 1: Launch PyCharm.

Step 2: Navigate to the "File" menu and select "Open."

Step 3: Locate the Manas directory and select it to open the project.

## Additional Data

Due to the extensive size of the dataset, you can access the complete artifact along with the dataset by downloading it from Google Drive via the following link: [Google Drive Link](https://drive.google.com/file/d/1x8jZ27Ho9tZ5H1bVOvOCQB0xpi6n3FnN/view?usp=sharing).

The provided dataset encompasses the following components:

* Python files essential for result replication.
* Both training and testing data subsets.
* Trained models, available as pkl files, for Original Manas, Transformed Manas, Manas, and Auto-Keras.
* Excel files containing error rate information for Original Manas, Transformed Manas, Manas, and Auto-Keras across different time periods. These errors can also be visualized using the provided trained model pkl files.

#### Model Mining
To mine the model, execute the following command:
```
python3 clone/model_mining/database_creation/model_collection.py
```  
The process of mining the model will require some time investment. Fortunately, we've already completed the download of the models from the autokeras/mined_model directory. These models are stored in the format of abstract neural networks (ANN), ready for further utilization.

#### Manas
To replicate Manas' outcomes, follow these steps:

**We use Blood Cell problem as an example.**

Step 1: Download the data from [Google Drive Link](https://drive.google.com/file/d/1x8jZ27Ho9tZ5H1bVOvOCQB0xpi6n3FnN/view?usp=sharing). Extract the zip file.

Step 2: copy the "Blood Cell.py", "xtest_bc.npy", "ytest_bc.npy", "xtrain_bc.npy", "ytrain_bc.npy" to "autokeras"
```
cd Manas Data/Blood Cell
cp xtest_bc.npy ytest_bc.npy xtrain_bc.npy ytrain_bc.npy autokeras/
cp Classification/Blood Cell.py autokeras/
```

Step 3: Run the file "Blood Cell.py" to get the result.
```
python3 Blood Cell.py
```

**Example to evaluate trained models:**

In the folder autokeras, there is a file "eval.py" which contains the code to run the trained models on the testing data.

Step 1: Replace "path" with correct path.

Step 2: Run the file "eval.py" 
```
python3 eval.py
```


