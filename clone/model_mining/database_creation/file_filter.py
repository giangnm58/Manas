import os
from clone.model_mining.database_creation.constant import Constant
import shutil
class ModelMining():
    def __init__(self):
        self.files = []
        self.unspfiles = []
        self.nn = []
        self.py = []
        self.keras_keyword = ['keras']
        self.path = "..\..\model_mining"
    def diff(self, l1, l2):
        return (list(set(l1) - set(l2)))

    # r=root, d=directories, f = files
    def file_filter(self,count):
        # if os.path.exists("\\clone\\model_mining\\database_creation\\py_file\\" + str(count)):
        #     print("xxxxxxxxxxxxxxxx")
        os.mkdir("..\..\keras_repo\clone_py_file\\" + str(count))

        for r, d, f in os.walk(self.path):
            for file in f:
                if '.py' in file:
                    f = open(os.path.join(r, file), encoding="ISO-8859-1")
                    temp_file = f.read()
                    for i in range(len(self.keras_keyword)):
                        if self.keras_keyword[i] in temp_file:
                            #print(os.path.join(r, file))
                            self.nn.append(os.path.join(r, file))
                            break
        for filelink in self.nn:
            nnfileopen = open(filelink.strip(), "r", encoding="ISO-8859-1")
            nnfile = nnfileopen.read()
            for i in range(len(Constant.unsupported_kerasapis)):
                if Constant.unsupported_kerasapis[i] in nnfile:
                    self.unspfiles.append(filelink)
                    #print(filelink,'')
                    break
            for i in range(len(Constant.kerasapis)):
                if Constant.kerasapis[i] in nnfile:
                    #print(filelink, 'yyyyyyyyyy')
                    self.files.append(filelink)
                    break
        pyfiles = open("py_with_model", "w", encoding="ISO-8859-1")
        # pyfiles.truncate(0)
        #os.mkdir("\\clone\\model_mining\\database_creation\\py_file\\" + str(count))
        self.py = self.diff(self.files, self.unspfiles)
        for f in self.py:
            #print('aaaaa')

            shutil.copy(f, "..\..\keras_repo\clone_py_file\\" + str(count))
            try:
                pyfiles.write(f.strip())
                pyfiles.write('\n')
            except UnicodeEncodeError:
                pass
        pyfiles.close()


# if __name__ == '__main__':
#     mm = ModelMining()
#     mm.file_filter()
