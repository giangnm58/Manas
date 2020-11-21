import os
import ast
from shutil import copy
from clone.model_mining.database_creation.utils import file_remover
from clone.model_mining.database_creation.constant import Constant
import os, os.path

def manas_database_creation():
    file_remover(Constant.model1_10, '*.txt')
    file_remover(Constant.model1_100, '*.txt')
    file_remover(Constant.model1_1000, '*.txt')
    file_remover(Constant.model3_10, '*.txt')
    file_remover(Constant.model3_100, '*.txt')
    file_remover(Constant.model3_1000, '*.txt')
    total_model = []
    model = []
    count = 0
    coordinate = []
    for r, d, f in os.walk('temp_processed_model/unprocess_model'):
        for file in f:
            try:
                if '.txt' in file:
                    f = open(os.path.join(r, file), encoding="ISO-8859-1")
                    channel = 0
                    out = 0
                    counter = 0
                    x = []
                    for line in f:
                        model.append(line)
                        counter += 1
                    f.seek(0)

                    for dictlayer in f:
                        if dictlayer == '\n':
                            break
                        dictlayer = ast.literal_eval(dictlayer)
                        #print(dictlayer, file)
                        if 'conv' in dictlayer['func'].lower():

                            if isinstance(list(dictlayer.values())[1], list):
                                if len(list(dictlayer.values())[1]) == 3:
                                    if list(dictlayer.values())[1][0] == 1:
                                        channel = 1
                                        #print(list(dictlayer.values())[1], file, 'aaaaaaaaaaaaaaaaaaaaaaa')
                                        coordinate.append(list(dictlayer.values())[1][1])
                                        coordinate.append(list(dictlayer.values())[1][2])
                                        x = list(dictlayer.values())[1]
                                        break
                                    if list(dictlayer.values())[1][0] == 3:
                                        channel = 3
                                        #print(list(dictlayer.values())[1], file, 'aaaaaaaaaaaaaaaaaaaaaaa')
                                        coordinate.append(list(dictlayer.values())[1][1])
                                        coordinate.append(list(dictlayer.values())[1][2])
                                        x = list(dictlayer.values())[1]
                                        break
                                    if list(dictlayer.values())[1][0] == 0:
                                        channel = 0
                                        #print(list(dictlayer.values())[1], file, 'aaaaaaaaaaaaaaaaaaaaaaa')
                                        x = list(dictlayer.values())[1]
                                        break
                    f.seek(0)
                    for dictlayer in reversed(list(f)):
                        dictlayer = ast.literal_eval(dictlayer)
                        if isinstance(dictlayer, dict):
                            if 'linear' in dictlayer['func'].lower():
                                if isinstance(list(dictlayer.values())[2], int):
                                    if 1 <= list(dictlayer.values())[2] <= 10:
                                        #print(list(dictlayer.values())[2], file, 'aaaaaaaaaaaaaaaaaaaaaaa')
                                        out = 10
                                        coordinate.append(list(dictlayer.values())[2])
                                        # if(channel != 0):
                                        #     print(list(dictlayer.values())[1])
                                        break
                                    if 10 < list(dictlayer.values())[2] <= 100:
                                        #print(list(dictlayer.values())[2], file, 'aaaaaaaaaaaaaaaaaaaaaaa')

                                        out = 100
                                        coordinate.append(list(dictlayer.values())[2])
                                        # if (channel != 0):
                                        #     print(list(dictlayer.values())[1])
                                        break
                                    if list(dictlayer.values())[2] > 100:
                                        #print(list(dictlayer.values())[2], file, 'aaaaaaaaaaaaaaaaaaaaaaa')

                                        out = 1000
                                        coordinate.append(list(dictlayer.values())[2])
                                        # if (channel != 0):
                                        #     print(list(dictlayer.values())[1])
                                        break
                                    if list(dictlayer.values())[2] == 0:
                                        #print(list(dictlayer.values())[2], file, 'aaaaaaaaaaaaaaaaaaaaaaa')

                                        out = 0
                                        break
                                else:
                                    out = 0
                    temp = model.copy()
                    f.close()

                    if temp in total_model:
                        count += 1
                        #print(file)
                    if temp not in total_model:
                        total_model.append(temp)
                        if counter > 10:
                            if channel == 3 and out == 10:
                                copy(os.path.join(r, file), Constant.model3_10)
                                f = open(os.path.join(Constant.model3_10, file), "a", encoding="ISO-8859-1")
                                f.write(str(coordinate))
                                f.close()
                                #print(file)

                            if channel == 1 and out == 10:
                                copy(os.path.join(r, file), Constant.model1_10)
                                f = open(os.path.join(Constant.model1_10, file), "a", encoding="ISO-8859-1")
                                f.write(str(coordinate))
                                f.close()
                                #print(file)

                            if channel == 3 and out == 100:
                                copy(os.path.join(r, file), Constant.model3_10)
                                f = open(os.path.join(Constant.model3_10, file), "a", encoding="ISO-8859-1")
                                f.write(str(coordinate))
                                f.close()
                                #print(file)

                            if channel == 1 and out == 100:
                                copy(os.path.join(r, file), Constant.model1_10)
                                f = open(os.path.join(Constant.model1_10, file), "a", encoding="ISO-8859-1")
                                f.write(str(coordinate))
                                f.close()
                                #print(file)


                            if channel == 3 and out == 1000:
                                copy(os.path.join(r, file), Constant.model3_10)
                                f = open(os.path.join(Constant.model3_10, file), "a", encoding="ISO-8859-1")
                                f.write(str(coordinate))
                                f.close()
                                #print(file)

                            if channel == 1 and out == 1000:
                                copy(os.path.join(r, file), Constant.model1_10)
                                f = open(os.path.join(Constant.model1_10, file), "a", encoding="ISO-8859-1")
                                f.write(str(coordinate))
                                f.close()
                                #print(file)
                    coordinate.clear()
                    model.clear()
            except (IndexError, AttributeError):
                #print('xxxxxxxxxxxxxxxxxxx')
                pass
    #print(count)

'''

def manas_database_creation():
    file_remover(Constant.model1_10, '*.txt')
    file_remover(Constant.model1_100, '*.txt')
    file_remover(Constant.model1_1000, '*.txt')
    file_remover(Constant.model3_10, '*.txt')
    file_remover(Constant.model3_100, '*.txt')
    file_remover(Constant.model3_1000, '*.txt')
    total_model = []
    model = []
    count = 0
    for r, d, f in os.walk('temp_processed_model/unprocess_model'):
        for file in f:
            if '.txt' in file:
                f = open(os.path.join(r, file), encoding="ISO-8859-1")
                channel = 0
                out = 0
                counter = 0
                for line in f:
                    model.append(line)
                    counter += 1
                f.seek(0)
                for dictlayer in reversed(list(f)):
                    dictlayer = ast.literal_eval(dictlayer)
                    if 'linear' in dictlayer['func'].lower():
                        if isinstance(list(dictlayer.values())[2], int):
                            if 1 <= list(dictlayer.values())[2] <= 10:
                                #print(list(dictlayer.values())[2], file, 'aaaaaaaaaaaaaaaaaaaaaaa')
                                out = 10
                                if(channel != 0):
                                    print(list(dictlayer.values())[1])
                                break
                            if 10 < list(dictlayer.values())[2] <= 100:
                                #print(list(dictlayer.values())[2], file, 'aaaaaaaaaaaaaaaaaaaaaaa')

                                out = 100

                                break
                            if list(dictlayer.values())[2] > 100:
                                #print(list(dictlayer.values())[2], file, 'aaaaaaaaaaaaaaaaaaaaaaa')

                                out = 1000
                                if (channel != 0):
                                    print(list(dictlayer.values())[1])
                                break
                            if list(dictlayer.values())[2] == 0:
                                #print(list(dictlayer.values())[2], file, 'aaaaaaaaaaaaaaaaaaaaaaa')

                                out = 0
                                break
                        else:
                            out = 0

                f.seek(0)
                for dictlayer in f:
                    if dictlayer == '\n':
                        break
                    dictlayer = ast.literal_eval(dictlayer)
                    if 'conv' in dictlayer['func'].lower():

                        if isinstance(list(dictlayer.values())[1], list):
                            if list(dictlayer.values())[1][0] == 1:
                                channel = 1
                                #print(list(dictlayer.values())[1], file, 'aaaaaaaaaaaaaaaaaaaaaaa')
                                if (out != 0):
                                    print(list(dictlayer.values())[1], out, file)
                                break
                            if list(dictlayer.values())[1][0] == 3:
                                channel = 3
                                #print(list(dictlayer.values())[1], file, 'aaaaaaaaaaaaaaaaaaaaaaa')
                                #if (out != 0):
                                    #print(list(dictlayer.values())[1], out, file)
                                break
                            if list(dictlayer.values())[1][0] == 0:
                                channel = 0
                                #print(list(dictlayer.values())[1], file, 'aaaaaaaaaaaaaaaaaaaaaaa')

                                break

                temp = model.copy()
                if temp in total_model:
                    count += 1
                if temp not in total_model:
                    total_model.append(temp)
                    if counter > 10:
                        if channel == 3 and out == 10:
                            copy(os.path.join(r, file), Constant.model3_10)
                            #print(file)

                        if channel == 1 and out == 10:
                            copy(os.path.join(r, file), Constant.model1_10)
                            #print(file)

                        if channel == 3 and out == 100:
                            copy(os.path.join(r, file), Constant.model3_100)
                            #print(file)

                        if channel == 1 and out == 100:
                            copy(os.path.join(r, file), Constant.model1_100)
                            #print(file)

                        if channel == 3 and out == 1000:
                            copy(os.path.join(r, file), Constant.model3_1000)
                            #print(file)

                        if channel == 1 and out == 1000:
                            copy(os.path.join(r, file), Constant.model1_1000)
                            #print(file)

                model.clear()
    #print(count)
'''
#manas_database_creation()