from clone.model_mining.controlflowgraph import ControlFlowGraph
from clone.keras_repo import repo_download
from clone.model_mining.database_creation.file_filter import ModelMining
from clone.model_mining.database_creation.database import manas_database_creation
from clone.model_mining.database_creation.utils import read_files, file_filtered, file_remover
from clone.model_mining.database_creation.constant import Constant
from clone.model_mining.database_creation.process_file import preprocessing
import os
def collect_model():
    repo_download.download()
    file_remover('temp_processed_model/unprocess_model', '*.txt')
    file_remover('unprocess_model', '*.txt')
    #file_remover('py_files', '*.txt')
    #pyfiles = open("py_with_model", "r", encoding="ISO-8859-1")
    cfgrun_file = "tempfiles/cfgrunning.py"
    count = 0
    #for py in pyfiles:
    for r, d, f in os.walk('..\..\keras_repo\clone_py_file'):
        for file in f:
            if '.py' in file:
                count += 1
                temp_file = open(r + "\\" + file, "r+", encoding="ISO-8859-1")
                cfgfile = open(cfgrun_file, "w", encoding="ISO-8859-1")
                cfgfile.truncate(0)
                cfgfile.write('import os\n')
                for line in temp_file:
                    cfgfile.write(line)
                cfgfile.close()
                cfg = ControlFlowGraph(Constant.kerasapis, "unprocess_model/model" + str(count) + ".txt")
                try:
                    cfg.parse_file("tempfiles/cfgrunning.py")
                except (TabError, SyntaxError, AttributeError, UnicodeEncodeError):
                    #print(r + "\\" + file, 'xxxxxxxxxxxxx')
                    pass
                #print(r + "\\" + file, count)
    file_filtered('unprocess_model', '.txt')
    preprocessing()
    manas_database_creation()
if __name__ == '__main__':
    collect_model()
