We use Pycharm to run the program conveniently.

Download Pycharm from https://www.jetbrains.com/pycharm/ and install Pycharm.

Step 1: open Pycharm

Step 2: go to File, then go to Open.

Step 3: choose the directory of Manas to open the project.


1. Model mining:
To mine the model run the file clone/model_mining/database_creation/model_collection.py
Mining model will take time, we already download the models located in autokeras/mined_model. The models is stored in form of abstract neural network (ANN).

2. Manas:

The data of Manas paper can be downloaded from: https://drive.google.com/file/d/1gibRjgVJmRUl_YorLmYtdRM4YhFUMxm5/view?usp=sharing

Data contains:

- Python files to reproduce the results. 

- Training and testing data.

- Trained models of Original Manas, Transformed Manas, Manas, and Auto-Keras in form of pkl files.

- Excel files for error rates of Original Manas, Transformed Manas, Manas, and Auto-Keras over time. These error can also be shown by using trained models (pkl files).

Example to reproduce Manas results:

We use Blood Cell problem as an example.

Step 1: Extract zip file.

Step 2: copy the "Blood Cell.py", "xtest_bc.npy", "ytest_bc.npy", "xtrain_bc.npy", "ytrain_bc.npy" to "autokeras"

Step 3: Run the file "Blood Cell.py" to get the result.

Example to evaluate trained models:

In the folder autokeras, there is a file "eval.py" which contains the code to run the trained models on the testing data.

Step 1: Replace "path" with correct path.

Step 2: Run the file "eval.py" 
