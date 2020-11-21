import os
import os
from clone.model_mining.database_creation.constant import Constant
import glob, os, os.path


def read_files(file_path, type):
    files = []
    for r, d, f in os.walk(file_path):
        for file in f:
            if type in file:
                files.append(os.path.join(r, file))
    return files

def file_remover(path, type):
    filelist = glob.glob(os.path.join(path, type))
    for f in filelist:
        os.remove(f)

def file_filtered(file_path, type):
    files = read_files(file_path, type)
    for file in files:
        f = open(file, 'r', encoding="ISO-8859-1")
        temp_file = f.read()
        f.close()
        if os.stat(file).st_size == 0:
            os.remove(file)
        for api in Constant.unsupported_kerasapis:
            if api in temp_file:
                try:
                    os.remove(file)
                except FileNotFoundError:
                    pass
                break

def valid_check(current_layer, last_layer):
    if current_layer == last_layer:
        return True
    else:
        return False