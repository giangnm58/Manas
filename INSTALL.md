
## Installation

For optimal program execution, we employ PyCharm, a user-friendly integrated development environment.

To acquire PyCharm, visit https://www.jetbrains.com/pycharm/ and proceed with the installation.

Here's a concise guide to getting started:

Step 1: Launch PyCharm.

Step 2: Navigate to the "File" menu and select "Open."

Step 3: Locate the Manas directory and select it to open the project.

1. Model mining:
To mine the model, execute the following command:
```
python3 clone/model_mining/database_creation/model_collection.py
```  
The process of mining the model will require some time investment. Fortunately, we've already completed the download of the models from the autokeras/mined_model directory. These models are stored in the format of abstract neural networks (ANN), ready for further utilization.

2. Manas:

The data of Manas paper can be downloaded from: https://drive.google.com/file/d/1x8jZ27Ho9tZ5H1bVOvOCQB0xpi6n3FnN/view?usp=sharing



We use Blood Cell problem as an example.

Step 1: Extract zip file.

Step 2: copy the "Blood Cell.py", "xtest_bc.npy", "ytest_bc.npy", "xtrain_bc.npy", "ytrain_bc.npy" to "autokeras"

Step 3: Run the file "Blood Cell.py" to get the result.

Example to evaluate trained models:

In the folder autokeras, there is a file "eval.py" which contains the code to run the trained models on the testing data.

Step 1: Replace "path" with correct path.

Step 2: Run the file "eval.py" 


## Data

Due to the extensive size of the dataset, you can access the complete artifact along with the dataset by downloading it from Google Drive via the following link: [Google Drive Link](https://drive.google.com/file/d/1x8jZ27Ho9tZ5H1bVOvOCQB0xpi6n3FnN/view?usp=sharing).

The provided dataset encompasses the following components:

* Python files essential for result replication.
* Both training and testing data subsets.
* Trained models, available as pkl files, for Original Manas, Transformed Manas, Manas, and Auto-Keras.
* Excel files containing error rate information for Original Manas, Transformed Manas, Manas, and Auto-Keras across different time periods. These errors can also be visualized using the provided trained model pkl files.
