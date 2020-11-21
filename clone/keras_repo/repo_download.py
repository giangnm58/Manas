import urllib.request, json
import time
import json
import os
from clone.model_mining.database_creation.file_filter import ModelMining
import os, shutil, stat

def handleError(func, path, exc_info):
    print('Handling Error for file ' , path)
    print(exc_info)
    # Check if file access issue
    if not os.access(path, os.W_OK):
       # Try to change the permision of file
       os.chmod(path, stat.S_IWUSR)
       # call the calling function again
       func(path)
def download():
    for year in [2015, 2016, 2017, 2018, 2019, 2020]:
        for j in range(1, 13):
            url = ''
            mins = 0
            while mins < 3:
                for i in range(mins * 10 + 1, mins * 10 + 11):
                    if i < 10 and j < 10:
                        url = 'https://api.github.com/search/repositories?q=keras+language:python+created:' +str(year)+'-0' + str(
                            j) + '-0' + str(i) + "&per_page=100"
                    if i < 10 and j >= 10:
                        url = 'https://api.github.com/search/repositories?q=keras+language:python+created:'+str(year)+'-' + str(
                            j) + '-0' + str(i) + "&per_page=100"
                    if i >= 10 and j < 10:
                        url = 'https://api.github.com/search/repositories?q=keras+language:python+created:'+str(year)+'-0' + str(
                            j) + '-' + str(i) + "&per_page=100"
                    if i >= 10 and j >= 10:
                        url = 'https://api.github.com/search/repositories?q=keras+language:python+created:'+str(year)+'-' + str(
                            j) + '-' + str(i) + "&per_page=100"
                    try:
                        with urllib.request.urlopen(url) as url1:
                            data = json.loads(url1.read().decode())
                        with open("keras" + str(year) + "_" + str(j) + '_' + str(i) + ".json", "w+") as outfile:
                            json.dump(data, outfile)
                    except urllib.error.HTTPError:
                        pass
                time.sleep(60)
                mins += 1




    clone_list = {}
    count = 0
    for r, d, f in os.walk(os.getcwd()):
        for file in f:
            if '.json' in file:
                json_file = open(os.path.join(r, file), encoding="ISO-8859-1")
                data = json.load(json_file)
                if 'items' in data:
                    for p in data['items']:
                        clone_list.update({p['clone_url']: p['stargazers_count']})
    list = sorted(clone_list.items(), key=lambda x: x[1], reverse=True)
    print(list)
    for repo in list:
        count += 1
        if 10000 >= count >= 0:
            os.system("git clone " + repo[0])
            mm = ModelMining()
            mm.file_filter(count)
            #print(repo[0].replace("https://github.com/", "").partition("/")[2].replace(".git", ""))
            try:
                shutil.rmtree(repo[0].replace("https://github.com/", "").partition("/")[2].replace(".git", ""), onerror=handleError)
            except FileNotFoundError:
                pass
#download()