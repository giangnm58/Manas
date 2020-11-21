import os
file1 = open("test.txt","r")
arr = []
for i in file1:
    arr.append(i.strip())
for r, d, f in os.walk('temp_processed_model/unprocess_model'):
    for file in f:
        if '.txt' in file:
            if file in arr:
                os.remove(os.path.join(r, file))