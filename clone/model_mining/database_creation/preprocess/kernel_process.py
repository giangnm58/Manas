def kernel_preprocess(dictlayer, tempdict, input, valid):
    if isinstance(dictlayer[input], int):
        tempdict.update({'kernel_size': [dictlayer[input]]})
    else:
        if len(dictlayer[input]) != 0:
            tempdict.update({'kernel_size': dictlayer[input]})
        else:
            valid = False

def strides_preprocess(dictlayer, tempdict, input):
    if isinstance(dictlayer[input], int):
        tempdict.update({'strides': [dictlayer[input]]})
    else:
        tempdict.update({'strides': dictlayer[input]})