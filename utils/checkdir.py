import os

def checkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


