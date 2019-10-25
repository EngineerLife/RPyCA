# File holds list of csv files to parse thru

from os import listdir
from os.path import isfile, join

# gets the list of csv's from directory
def getFile(day='', typ=False):
    days = ['Monday','Tuesday','Wednesday','Thursday','Friday']
    csvList = []
    cdir = "../../MachineLearningCVE/"
    if typ:
        cdir = "../../TrafficLabelling/"
    # order list by day of week (M->F)
    for d in days:
        for f in listdir(cdir):
            if isfile(join(cdir, f)) and d in f and day in f:
                csvList.append(cdir+f)

    return csvList
