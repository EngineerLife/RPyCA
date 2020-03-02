# Contains methods to handle files in the data set
import re
import numpy as np
import pandas as pd
import time, csv
from .logger import logMsg
from os import listdir, chdir, curdir, getcwd
from os.path import isfile, join

# function for handling IP addresses;
# separates the bytes into their own features
# INPUT: ip addr
# OUTPUT: 
def splitIP(ipAddr):
    sepIP = ipAddr.split(".")
#    print(sepIP)
#    exit(0)
    return sepIP

# TODO account for headers, labels, orientation
# loads a list of files and extracts/forms contents
# returns data wanted for X and the labels of those columns
'''
def loadFile(name, header, labelLoc, rowClmn, skip=[]):
    mat, featLabels, labels = [], [], []
    count = 0
    with open(name) as fin:
#        print(len(fin.readlines()))    # counts line in file
        for line in fin:
            lineData = line.split(",")

            # creates the y array (label data)
            if not header:  # skips header
                if not "BENIGN" in lineData[labelLoc]:
                    labels.append(1)
                else:
                    labels.append(0)

            # NOTE UNB data set gets all data, except for Flow ID, SOURCE IP ADDR, Timestamp, and label
            temp = []
            for i in range(len(lineData)):
                ld = lineData[i]
                # skips iteration
                if i in skip:
                    continue
                # only runs for header line in file
                if header:
                    if "IP" in ld:
                        ipClmn = i
                        for _ in range(4): 
                            featLabels.append(ld)   # for distiguishing what feature is per label
                    else:
                        featLabels.append(ld)
                else:
                    if i == ipClmn:
                        ld = splitIP(ld)    # only gets destination ip to split
                        for ip in ld:
                            temp.append(ip)
                    else:
                        temp.append(ld)
            if header:
                logMsg(1,"FEATURES USED %s" % str(featLabels))
            else:
                mat.append(temp)
            count += 1
            header = 0  # accounted for header, if there was one
    return np.matrix(mat), featLabels, labels 
'''

def load(name, labelName, skip=[]):
    df = pd.read_csv(name)
    df.columns = df.columns.str.replace(' ', '')
    skip = ['FlowID']
#    labelName = 'Label'
    labels = df[labelName]
#    to_drop = skip.append(labelName)
    df = df.drop(columns=['FlowID', 'SourceIP', 'Timestamp', 'Label'])
#    print(df.columns)
    return df, df.columns, labels

# saves matrix data to file
def save(data, fileName):
    fn = "helperFiles/files/" + fileName + ".csv"
#    f = open(fn, "w")
#    print("SAVING", fn)
    logMsg(1,"Saving final X matrix to %s" % fn)
#    f = open(fileName, "w")
#    for i in data:
#        f.write(str(i[0])+"\n")
#    f.close()
    # NOTE needs to be a pandas dataframe
    data.to_csv(fn, index=False)

# takes a string that's a list and converts it to a list
def toList(string, integer=True):
    splitStr = re.split('\[|,|\]|',string)
    while "" in splitStr:
        splitStr.remove("")
    if integer:
        return [int(i) for i in splitStr]
    return [re.sub(r'[^\w]', '', i) for i in splitStr]
