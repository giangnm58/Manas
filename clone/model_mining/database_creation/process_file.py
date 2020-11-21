import os
import ast
from clone.model_mining.database_creation.utils import read_files, valid_check
'''
def preprocessing():
    files = read_files('unprocess_model', '.txt')

    # Conv2D = {'filters': 'int', 'kernel_size': 'int/tuple/list', 'strides': 'tuple/list', 'activation': 'str'}
    for file in files:
        try:
            # checker = 0
            valid = True
            last_layer = None
            model_check = False
            count_conv = 0
            input_value = 0
            model = open(file, encoding="ISO-8859-1")
            proccessed_model = []
            for layer in model:
                dictlayer = ast.literal_eval(layer)
                tempdict = {}
                if dictlayer['func'] == 'Input':
                    if len(list(dictlayer.values())) > 1:
                        if isinstance(list(dictlayer.values())[1], list):
                            #if len(list(dictlayer.values())[1]) > 0:
                            if len(list(dictlayer.values())[1]) == 0:
                                list(dictlayer.values())[1].sort()
                                input_value = list(dictlayer.values())[1]
                                print(list(dictlayer.values())[1], file, 'xxxxxxxxxx')
                            else:
                                valid = False
                                break
                        else:
                            valid = False
                            break
                    else:
                        valid = False
                        break
                        #input_value = 0
                    model_check = valid_check('Input', last_layer)
                    last_layer = 'Input'

                # Add Conv2d layer
                # Conv2D = {'filters': 'int', 'kernel_size': 'int/tuple/list', 'strides': 'tuple/list', 'activation': 'str'}
                if dictlayer['func'] == 'Conv2D':
                    # Check valid layer
                    count_conv += 1
                    tempdict.update({'func': "Conv2D"})
                    if 'arg1' or 'filters' in dictlayer:
                        valid = True
                    else:
                        valid = False
                        #checker = 1
                        break
                    if 'arg2' or 'kernel_size' in dictlayer:
                        valid = True
                    else:
                        valid = False
                        #checker = 2
                        break

                    # add input, output values
                    if isinstance(list(dictlayer.values())[1], int):
                        if 'input_shape' in dictlayer:
                            if len(dictlayer['input_shape']) > 0:
                                print(dictlayer['input_shape'])
                                dictlayer['input_shape'].sort()
                                #tempdict.update({'arg1': dictlayer['input_shape'][0]})
                                tempdict.update({'arg1': dictlayer['input_shape']})
                                tempdict.update({'arg2': list(dictlayer.values())[1]})
                                print(dictlayer['input_shape'],file, 'xxxxxxxxxx')
                            else:
                                tempdict.update({'arg1': 0})
                                tempdict.update({'arg2': list(dictlayer.values())[1]})
                        else:
                            tempdict.update({'arg1': input_value})
                            tempdict.update({'arg2': list(dictlayer.values())[1]})
                    else:
                        valid = False

                    # add kernel_size value
                    if 'arg2' in dictlayer:
                        if isinstance(dictlayer['arg2'], int):
                            tempdict.update({'kernel_size': [dictlayer['arg2']]})
                        else:
                            if len(dictlayer['arg2']) != 0:
                                tempdict.update({'kernel_size': dictlayer['arg2']})
                            else:
                                valid = False
                    elif 'kernel_size' in dictlayer:
                        if isinstance(dictlayer['kernel_size'], int):
                            tempdict.update({'kernel_size': [dictlayer['kernel_size']]})
                        else:
                            if len(dictlayer['kernel_size']) != 0:
                                tempdict.update({'kernel_size': dictlayer['kernel_size']})
                            else:
                                valid = False
                    # add stride value
                    if 'strides' in dictlayer:
                        if isinstance(dictlayer['strides'], int):
                            tempdict.update({'strides': [dictlayer['strides']]})
                        else:
                            tempdict.update({'strides': dictlayer['strides']})
                    if 'padding' in dictlayer:
                        if dictlayer['padding'] == 'valid':
                            tempdict.update({'padding': 0})


                    proccessed_model.append(tempdict)

                    # extract activation and add new layer
                    if 'activation' in dictlayer:
                        proccessed_model.append({'func': dictlayer['activation']})
                    input_value = list(dictlayer.values())[1]
                    #print(list(dictlayer.values()),'xxxxxxxxxxx')
                    model_check = valid_check('Conv2D', last_layer)
                    last_layer = 'Conv2D'

                # Add Convolution2d layer
                if dictlayer['func'] == 'Convolution2D':

                    # Check valid layer
                    count_conv += 1
                    tempdict.update({'func': "Conv2D"})
                    if 'arg1' or 'nb_filter' in dictlayer:
                        valid = True
                    else:
                        valid = False
                        #checker = 3
                        break
                    if 'arg2' or 'nb_row' in dictlayer:
                        valid = True
                    else:
                        valid = False
                        #checker = 4
                        break
                    if 'arg3' or 'nb_col' in dictlayer:
                        valid = True
                    else:
                        valid = False
                        #checker = 5
                        break

                    # add input, output values
                    if 'input_shape' in dictlayer:
                        if len(dictlayer['input_shape']) > 0:
                            dictlayer['input_shape'].sort()
                            #tempdict.update({'arg1': dictlayer['input_shape'][0]})
                            tempdict.update({'arg1': dictlayer['input_shape']})
                            tempdict.update({'arg2': list(dictlayer.values())[1]})
                            print(dictlayer['input_shape'], file, 'xxxxxxxxxx')
                        else:
                            tempdict.update({'arg1': 0})
                            tempdict.update({'arg2': list(dictlayer.values())[1]})
                    else:
                        tempdict.update({'arg1': input_value})
                        tempdict.update({'arg2': list(dictlayer.values())[1]})

                    # add kernel_size value
                    if isinstance(list(dictlayer.values())[2], int):
                        tempdict.update({'kernel_size': [list(dictlayer.values())[2]]})
                    else:
                        valid = False
                        break
                    
                    if isinstance(list(dictlayer.values())[2], int):
                        if 'arg2' in dictlayer:
                            if isinstance(dictlayer['arg2'], int):
                                tempdict.update({'kernel_size': [dictlayer['arg2']]})
                            else:
                                tempdict.update({'kernel_size': dictlayer['arg2']})
                        elif 'kernel_size' in dictlayer:
                            if isinstance(dictlayer['kernel_size'], int):
                                tempdict.update({'kernel_size': [dictlayer['kernel_size']]})
                            else:
                                tempdict.update({'kernel_size': dictlayer['kernel_size']})
                        elif 'nb_row' in dictlayer:
                            if isinstance(dictlayer['nb_row'], int):
                                tempdict.update({'kernel_size': [dictlayer['nb_row']]})
                            else:
                                tempdict.update({'kernel_size': dictlayer['nb_row']})
                    else:
                        valid = False
                        break
                    
                    # add stride value
                    if 'subsample' in dictlayer:
                        tempdict.update({'strides': dictlayer['subsample']})

                    if 'border_mode' in dictlayer:
                        if dictlayer['border_mode'] == 'valid':
                            tempdict.update({'padding': 0})
                    proccessed_model.append(tempdict)

                    # extract activation and add new layer
                    if 'activation' in dictlayer:
                        proccessed_model.append({'func': dictlayer['activation']})
                        last_layer = dictlayer['activation']
                    input_value = list(dictlayer.values())[1]
                    model_check = valid_check('Convolution2D', last_layer)
                    last_layer = 'Convolution2D'

                # add Dense layer
                if dictlayer['func'] == 'Dense':
                    # if len(list(dictlayer.values())) < 2:
                    # valid = False
                    # checker = 6
                    # break
                    # add input output value
                    tempdict.update({'func': 'linear'})
                    if isinstance(input_value, int) and input_value != 0:
                        tempdict.update({'arg1': input_value})
                    else:
                        valid =False
                        break
                    if len(list(dictlayer.values())) > 1:
                        if isinstance(list(dictlayer.values())[1], int):
                            tempdict.update({'arg2': list(dictlayer.values())[1]})
                        else:
                            valid = False
                            break
                    else:
                        tempdict.update({'arg2': 0})
                    proccessed_model.append(tempdict)

                    # extract activation and add new layer
                    if 'activation' in dictlayer:
                        proccessed_model.append({'func': dictlayer['activation']})
                        last_layer = dictlayer['activation']
                    if len(list(dictlayer.values())) > 1:
                        input_value = list(dictlayer.values())[1]
                    else:
                        valid = False
                        break
                        #input_value = 0

                # add MaxPooling2D layer
                if dictlayer['func'] == 'MaxPooling2D':
                    tempdict.update({'func': 'MaxPool2d'})
                    if 'arg1' in dictlayer:
                        if isinstance(dictlayer['arg1'], int):
                            tempdict.update({'pool_size': [dictlayer['arg1']]})
                        else:
                            if len(dictlayer['arg1']) != 0:
                                tempdict.update({'pool_size': dictlayer['arg1']})
                    elif 'pool_size' in dictlayer:
                        if isinstance(dictlayer['pool_size'], int):
                            tempdict.update({'pool_size': [dictlayer['pool_size']]})
                        else:
                            if len(dictlayer['pool_size']) != 0:
                                tempdict.update({'pool_size': dictlayer['pool_size']})

                    if 'arg2' in dictlayer:
                        if isinstance(dictlayer['arg2'], int):
                            tempdict.update({'strides': [dictlayer['arg2']]})
                        else:
                            if len(dictlayer['arg2']) != 0:
                                tempdict.update({'strides': dictlayer['arg2']})
                    elif 'strides' in dictlayer:
                        if isinstance(dictlayer['strides'], int):
                            tempdict.update({'strides': [dictlayer['strides']]})
                        else:
                            if len(dictlayer['strides']) != 0:
                                tempdict.update({'strides': dictlayer['strides']})

                    proccessed_model.append(tempdict)
                    model_check = valid_check('MaxPooling2D', last_layer)
                    last_layer = 'MaxPooling2D'

                # add Dropout layer
                if dictlayer['func'] == 'Dropout':
                    if len(list(dictlayer.values())) < 2:
                        valid = False
                        #checker = 7
                        break
                    tempdict.update({'func': dictlayer['func']})
                    tempdict.update({'arg1': list(dictlayer.values())[1]})
                    proccessed_model.append(tempdict)
                    model_check = valid_check('Dropout', last_layer)
                    last_layer = 'Dropout'

                # add Flatten layer
                if dictlayer['func'] == 'Flatten':
                    proccessed_model.append({'func': dictlayer['func']})
                    model_check = valid_check('Flatten', last_layer)
                    last_layer = 'Flatten'

                # add Activation layer
                if dictlayer['func'] == 'Activation':
                    if len(list(dictlayer.values())) < 2:
                        valid = False
                        #checker = 8
                        break
                    proccessed_model.append({'func': list(dictlayer.values())[1]})
                    model_check = valid_check('Activation', last_layer)
                    last_layer = 'Activation'

                # add BatchNormalization layer
                if dictlayer['func'] == 'BatchNormalization':
                    proccessed_model.append({'func': 'BatchNorm2d'})
                    model_check = valid_check('BatchNormalization', last_layer)
                    last_layer = 'BatchNormalization'

                # add AveragePooling2D layer
                if dictlayer['func'] == 'AveragePooling2D':
                    proccessed_model.append({'func': 'AvgPool2d'})
                    model_check = valid_check('AveragePooling2D', last_layer)
                    last_layer = 'AveragePooling2D'

                if dictlayer['func'] in ['compile', 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']:
                    proccessed_model.append(dictlayer)
                    break

            # if dictlayer['func'] == 'GlobalAveragePooling2D':
            model.close()
            print(file, valid, count_conv)
            if valid == True and count_conv > 0 and model_check == False:
                model_writer = open('temp_processed_model/' + file, 'w', encoding="ISO-8859-1")
                for i in range(len(proccessed_model)):
                    model_writer.write(str(proccessed_model[i]) + '\n')
                model_writer.close()
        except (IndexError, TypeError):
            pass
'''
import os
import ast
from clone.model_mining.database_creation.utils import read_files, valid_check

def preprocessing():
    files = read_files('unprocess_model', '.txt')

    # Conv2D = {'filters': 'int', 'kernel_size': 'int/tuple/list', 'strides': 'tuple/list', 'activation': 'str'}
    for file in files:
        try:
            valid = True
            count_conv = 0
            count_dense = 0
            #checker = 0
            last_layer = None
            model_check = False
            model = open(file, encoding="ISO-8859-1")
            # print(file)
            # try:
            input_value = 0
            proccessed_model = []
            for layer in model:

                dictlayer = ast.literal_eval(layer)
                tempdict = {}

                if dictlayer['func'] == 'Input':
                    # print(file)
                    # print(list(dictlayer.values())[1])
                    if len(list(dictlayer.values())) > 1:
                        if isinstance(list(dictlayer.values())[1], list):
                            if len(list(dictlayer.values())[1]) > 0:
                                list(dictlayer.values())[1].sort()
                                #input_value = list(dictlayer.values())[1][0]
                                input_value = list(dictlayer.values())[1]
                                #print(list(dictlayer.values())[1],file,'xxxxxxxxxx')
                            else:
                                valid = False
                                break
                                #input_value = 0
                        else:
                            valid = False
                            break
                            #input_value = 0
                    else:
                        valid = False
                        break
                        #input_value = 0
                    model_check = valid_check('Input', last_layer)
                    last_layer = 'Input'

                # Add Conv2d layer
                # Conv2D = {'filters': 'int', 'kernel_size': 'int/tuple/list', 'strides': 'tuple/list', 'activation': 'str'}
                if dictlayer['func'] == 'Conv2D':
                    # Check valid layer
                    count_conv += 1
                    tempdict.update({'func': "Conv2D"})
                    if 'arg1' in dictlayer or 'filters' in dictlayer:
                        valid = True
                    else:
                        valid = False
                        #checker = 1
                        break
                    if ('arg2' in dictlayer or 'kernel_size' in dictlayer) :
                        print(dictlayer,'kkkkkkkkkkk')
                        valid = True
                    else:
                        valid = False
                        #checker = 2
                        break

                    # add input, output values
                    if isinstance(list(dictlayer.values())[1], int):
                        if 'input_shape' in dictlayer:
                            if len(dictlayer['input_shape']) > 0:
                                print(dictlayer['input_shape'])
                                dictlayer['input_shape'].sort()
                                #tempdict.update({'arg1': dictlayer['input_shape'][0]})
                                tempdict.update({'arg1': dictlayer['input_shape']})
                                tempdict.update({'arg2': list(dictlayer.values())[1]})
                                print(dictlayer['input_shape'],file, 'xxxxxxxxxx')
                            else:
                                tempdict.update({'arg1': 0})
                                tempdict.update({'arg2': list(dictlayer.values())[1]})
                        else:
                            tempdict.update({'arg1': input_value})
                            tempdict.update({'arg2': list(dictlayer.values())[1]})
                    else:
                        valid = False

                    # add kernel_size value
                    if 'arg2' in dictlayer:
                        if isinstance(dictlayer['arg2'], int):
                            tempdict.update({'kernel_size': [dictlayer['arg2']]})
                        else:
                            if len(dictlayer['arg2']) != 0:
                                tempdict.update({'kernel_size': dictlayer['arg2']})
                            else:
                                valid = False
                    elif 'kernel_size' in dictlayer:
                        if isinstance(dictlayer['kernel_size'], int):
                            tempdict.update({'kernel_size': [dictlayer['kernel_size']]})
                        else:
                            if len(dictlayer['kernel_size']) != 0:
                                tempdict.update({'kernel_size': dictlayer['kernel_size']})
                            else:
                                valid = False
                    # add stride value
                    if 'strides' in dictlayer:
                        if isinstance(dictlayer['strides'], int):
                            tempdict.update({'strides': [dictlayer['strides']]})
                        else:
                            tempdict.update({'strides': dictlayer['strides']})

                    proccessed_model.append(tempdict)

                    # extract activation and add new layer
                    if 'activation' in dictlayer:
                        if dictlayer['activation'] == 'linear':
                            proccessed_model.append({'func': 'relu'})
                        else:
                            proccessed_model.append({'func': dictlayer['activation']})
                    input_value = list(dictlayer.values())[1]
                    #print(list(dictlayer.values()),'xxxxxxxxxxx')
                    model_check = valid_check('Conv2D', last_layer)
                    last_layer = 'Conv2D'

                # Add Convolution2d layer
                if dictlayer['func'] == 'Convolution2D':

                    # Check valid layer
                    count_conv += 1
                    tempdict.update({'func': "Conv2D"})
                    if 'arg1' in dictlayer or 'nb_filter' in dictlayer:
                        valid = True
                    else:
                        valid = False
                        #checker = 3
                        break
                    if 'arg2' in dictlayer or 'nb_row' in dictlayer:
                        valid = True
                    else:
                        valid = False
                        #checker = 4
                        break
                    if 'arg3' in dictlayer or 'nb_col' in dictlayer:
                        valid = True
                    else:
                        valid = False
                        #checker = 5
                        break

                    # add input, output values
                    if 'input_shape' in dictlayer:
                        if len(dictlayer['input_shape']) > 0:
                            dictlayer['input_shape'].sort()
                            #tempdict.update({'arg1': dictlayer['input_shape'][0]})
                            tempdict.update({'arg1': dictlayer['input_shape']})
                            tempdict.update({'arg2': list(dictlayer.values())[1]})
                            print(dictlayer['input_shape'], file, 'xxxxxxxxxx')
                        else:
                            tempdict.update({'arg1': 0})
                            tempdict.update({'arg2': list(dictlayer.values())[1]})
                    else:
                        tempdict.update({'arg1': input_value})
                        tempdict.update({'arg2': list(dictlayer.values())[1]})

                    # add kernel_size value
                    if isinstance(list(dictlayer.values())[2], int):
                        tempdict.update({'kernel_size': [list(dictlayer.values())[2]]})
                    else:
                        valid = False
                        break
                    '''
                    if isinstance(list(dictlayer.values())[2], int):
                        if 'arg2' in dictlayer:
                            if isinstance(dictlayer['arg2'], int):
                                tempdict.update({'kernel_size': [dictlayer['arg2']]})
                            else:
                                tempdict.update({'kernel_size': dictlayer['arg2']})
                        elif 'kernel_size' in dictlayer:
                            if isinstance(dictlayer['kernel_size'], int):
                                tempdict.update({'kernel_size': [dictlayer['kernel_size']]})
                            else:
                                tempdict.update({'kernel_size': dictlayer['kernel_size']})
                        elif 'nb_row' in dictlayer:
                            if isinstance(dictlayer['nb_row'], int):
                                tempdict.update({'kernel_size': [dictlayer['nb_row']]})
                            else:
                                tempdict.update({'kernel_size': dictlayer['nb_row']})
                    else:
                        valid = False
                        break
                    '''
                    # add stride value
                    if 'subsample' in dictlayer:
                        tempdict.update({'strides': dictlayer['subsample']})
                    proccessed_model.append(tempdict)

                    # extract activation and add new layer
                    if 'activation' in dictlayer:
                        if dictlayer['activation'] == 'linear':
                            proccessed_model.append({'func': 'relu'})
                        else:
                            proccessed_model.append({'func': dictlayer['activation']})
                        last_layer = dictlayer['activation']
                    input_value = list(dictlayer.values())[1]
                    model_check = valid_check('Convolution2D', last_layer)
                    last_layer = 'Convolution2D'

                # add Dense layer
                if dictlayer['func'] == 'Dense':
                    count_dense += 1
                    if len(list(dictlayer.values())) < 2:
                        valid = False
                        break
                    # checker = 6
                    # break
                    # add input output value
                    tempdict.update({'func': 'linear'})
                    if isinstance(input_value, int) and input_value != 0:
                        tempdict.update({'arg1': input_value})
                    else:
                        valid =False
                        break
                    if len(list(dictlayer.values())) > 1:
                        if isinstance(list(dictlayer.values())[1], int):
                            tempdict.update({'arg2': list(dictlayer.values())[1]})
                        else:
                            valid = False
                            break
                    else:
                        tempdict.update({'arg2': 0})
                    proccessed_model.append(tempdict)

                    # extract activation and add new layer
                    if 'activation' in dictlayer:
                        if dictlayer['activation'] == 'linear':
                            proccessed_model.append({'func': 'relu'})
                        else:
                            proccessed_model.append({'func': dictlayer['activation']})
                        last_layer = dictlayer['activation']
                    if len(list(dictlayer.values())) > 1:
                        input_value = list(dictlayer.values())[1]
                    else:
                        valid = False
                        break
                        #input_value = 0

                # add MaxPooling2D layer
                if dictlayer['func'] == 'MaxPooling2D':
                    proccessed_model.append({'func': 'MaxPool2d'})
                    model_check = valid_check('MaxPooling2D', last_layer)
                    last_layer = 'MaxPooling2D'

                # add Dropout layer
                if dictlayer['func'] == 'Dropout':
                    if len(list(dictlayer.values())) < 2:
                        valid = False
                        #checker = 7
                        break
                    tempdict.update({'func': dictlayer['func']})
                    tempdict.update({'arg1': list(dictlayer.values())[1]})
                    proccessed_model.append(tempdict)
                    model_check = valid_check('Dropout', last_layer)
                    last_layer = 'Dropout'

                # add Flatten layer
                if dictlayer['func'] == 'Flatten':
                    proccessed_model.append({'func': dictlayer['func']})
                    model_check = valid_check('Flatten', last_layer)
                    last_layer = 'Flatten'

                # add Activation layer
                if dictlayer['func'] == 'Activation':
                    if len(list(dictlayer.values())) < 2:
                        valid = False
                        #checker = 8
                        break
                    if list(dictlayer.values())[1] == 'linear':
                        list(dictlayer.values())[1] = 'relu'
                    proccessed_model.append({'func': list(dictlayer.values())[1]})
                    model_check = valid_check('Activation', last_layer)
                    last_layer = 'Activation'

                # add BatchNormalization layer
                if dictlayer['func'] == 'BatchNormalization':
                    proccessed_model.append({'func': 'BatchNorm2d'})
                    model_check = valid_check('BatchNormalization', last_layer)
                    last_layer = 'BatchNormalization'

                # add AveragePooling2D layer
                if dictlayer['func'] == 'AveragePooling2D':
                    proccessed_model.append({'func': 'AvgPool2d'})
                    model_check = valid_check('AveragePooling2D', last_layer)
                    last_layer = 'AveragePooling2D'

                if dictlayer['func'] in ['compile', 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']:
                    proccessed_model.append(dictlayer)
                    break

            # if dictlayer['func'] == 'GlobalAveragePooling2D':
            model.close()
            print(count_dense)
            #if valid == True and count_conv > 0 and model_check == False:
            if count_conv == 0:
                model_writer = open('temp_processed_model/' + file, 'w', encoding="ISO-8859-1")
                for i in range(len(proccessed_model)):
                    model_writer.write(str(proccessed_model[i]) + '\n')
                model_writer.close()
        except (IndexError, TypeError, SyntaxError, ValueError):
            pass

preprocessing()
#preprocessing()
