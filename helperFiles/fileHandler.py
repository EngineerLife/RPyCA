# Contains methods to handle files in the data set
import numpy as np
import time, csv
from .logger import logMsg
from os import listdir, chdir, curdir, getcwd
from os.path import isfile, join

#
#
# XXX MIT LLDOS dataset functions
#
#
# Gets the data from txt file of Collapsed PCAP
#   Input: String of name of file to extract data from
#   Output: Matrix with each row in form of [Src IP, Dest IP, Protocol, length, Src Port, Dest Port]
# NOTE may error if there are duplicate lines 
def getLLDOSData(name):
    count = 0
    extract, foundPort = False, False
    info, matrix = [], []
    with open(name) as fin:
        for line in fin:
            if extract:
                linfo = line.split()
                info = linfo[2:6]
                extract = False
            if "Src Port:" in line:
                pinfo = line.split(",")
                srcPort = pinfo[1].split(": ")[1]
                destPort = pinfo[2].split(": ")[1]
                info.append(srcPort)
                info.append(destPort)
                foundPort = True
            if "No.     Time" in line:
                # check if port found
                if not foundPort and count > 0:
                    info.append(None)
                    info.append(None)
                    matrix.append(info)
                elif count > 0:
                    matrix.append(info)
                count += 1
                info = []
                extract, foundPort = True, False
#    print("NUMBER OF LINES: ",count)    # should be 347987 for LLS_DDOS_2.0.2-inside
    return np.matrix(matrix)

# gets the labels for each packet row
#   Input: String of name of file to extract data from
#   Output: Dictionary of phases and the attack indices in the LLDOS inside file
def getLLDOSLabels(name):
    directory = "inside/"
    phaseFiles = ["phase-1-inside", "phase-2-inside", "phase-3-inside"]
    phaseDict = {
        "P1": [],
        "P2": [],
        "P3": []
        }
    c = 0
    # collect info of attacks
    for phase in phaseFiles:
        c += 1
        path = directory + phase
        with open(path) as fil:
            get = False
            attacks = []
            for line in fil:
                if get:
                    attacks.append(line.split()[2:7])
                    get = False
                if "No.     Time" in line:
                    get = True
            p = "P" + str(c)
            phaseDict[p] = attacks

    points = {
        "P1": [],
        "P2": [],
        "P3": []
        }
    lc = 0  # NOTE needs to be at 0 else the graphs will be off by 1. When manually reading file, add 1 (file indexes lines at 1)
    # compare attack data to data set
    with open(name) as data:
        for line in data:
            conLine = "".join(line.split())
            if not conLine:     # newline, skip
                continue
            if conLine[0].isdigit() and not conLine[0] == "0":    # data we want
                noPkt = line.split()[0]
#                print(conLine[0], "  a.k.a. = ", noPkt, "<-#   lc->", lc)
                for d in phaseDict:     # iterate thru phases
                    atks = phaseDict.get(d)
                    for a in atks:      # iterate thru attacks in phase
                        atkData = ''.join(map(str, a))
                        if atkData in conLine:
                            val = points.get(d)
                            if not lc in val:  # record line
#                                print("FOUND FIRST MAL @ ",lc,"\nmal stuff item: ",a,"\nfile line: ",line)
                                val.append(lc)
                                points[d] = val
                lc += 1     # only counts lines that we collect data from!!!!
    return points


# reads the label data from the file given and converts them to lists
#   Input: String of name of file to extract data from
#   Output: 3 lists for the 3 phases
def listLLDOSLabels(name):
    malPkts1, malPkts2, malPkts3 = [], [], []
    e1, e2, e3 = False, False, False
    with open(name) as fin:
        for line in fin:
            if not line or "*" in line:
                continue
            
            if "P1" in line:
                e1 = True
                e2, e3 = False, False
                continue
            elif "P2" in line:
                e2 = True
                e1, e3 = False, False
                continue
            elif "P3" in line:
                e3 = True
                e1, e2 = False, False
                continue
            line = "".join(line.split())
            if e1:
                malPkts1.append(int(line))
            elif e2:
                malPkts2.append(int(line))
            elif e3:
                malPkts3.append(int(line))
    return malPkts1, malPkts2, malPkts3

####
# Create Y labels
####
# creates labels for data set
#   Input: list of attak point indexes
#           length of data
#   Output: list of labels [0, 1] 0 being benign and 1 being malicious
def createY(lenData, atkPnts):
    y = []
    j = 0
    for i in range(lenData):
        if j < len(atkPnts) and i == atkPnts[j]:     # NOTE we can do this bc atkPnts are in numerical order
            y.append(1)
#            y.append("attack")
            j += 1
        else:
            y.append(0)
#            y.append("normal")
    return y
#    return np.flip(y)



#
#
# XXX UNB main thesis dataset functions
#
#

# function for handling IP addresses;
# separates the bytes into their own features
# INPUT: ip addr
# OUTPUT: 
def splitIP(ipAddr):
    sepIP = ipAddr.split(".")
#    print(sepIP)
#    exit(0)
    return sepIP

# gets the list of csv's from directory
# eg: testing = getUNBFile(files, True)[1]
def getUNBFile(day='', typ=False):
    days = ['Monday','Tuesday','Wednesday','Thursday','Friday']
    csvList = []
    cdir = "MachineLearningCVE/"
    if typ:
        cdir = "datasets/TrafficLabelling/"
    # order list by day of week (M->F)
    for d in days:
        for f in listdir(cdir):
            if isfile(join(cdir, f)) and d.lower() in f.lower() and day.lower() in f.lower():
                csvList.append(cdir+f)

    return csvList


# loads a list of files and extracts/forms contents
# returns data wanted for X and the labels of those columns
def loadUNBFile(name):
    mat, featLabels = [], []
    count = 0
    with open(name) as fin:
#        print(len(fin.readlines()))    # counts line in file
        for line in fin:
            lineData = line.split(",")
#            indexes = [1,2,3,4,5,46]   # contains IP's
#            indexes = [2,4,5,46]
#            indexes = [2,4,5]  # gets all data, except for Flow ID, SOURCE IP ADDR, Timestamp, and label
            indexes = [2,3,4,5]  # gets all data, except for Flow ID, SOURCE IP ADDR, Timestamp, and label
            for x in range(7,84):
                indexes.append(x)

            temp = []
            for i in indexes:
                ld = lineData[i]
                if count == 0:
                    if "IP" in ld:
                        for i in range(4):
                            featLabels.append(ld)
                    else:
                        featLabels.append(ld)
                else:
                    if i == 3:
                        ld = splitIP(ld)    # only gets destination ip to split
                        for ip in ld:
                            temp.append(ip)
                    else:
                        temp.append(ld)
            if count == 0:
                logMsg(1,"FEATURES USED %s" % str(featLabels))
            else:
                mat.append(temp)
            count += 1
    return np.matrix(mat), featLabels

# FOR MAIN THESIS
# This is pretty similar as what is needed for CreateY
def loadUNBLabels(filename):
    malPkts = []
    first = True
    with open(filename) as fin:
        for line in fin:
            if not first:   # skips header
                lineInfo = line.split(",")
                if not "BENIGN" in lineInfo[84]:
                    malPkts.append(1)
                else:
                    malPkts.append(0)
            first = False
    return malPkts

# saves matrix data to file
def save(data, fileName):
    fn = "helperFiles/files/" + fileName
    f = open(fn, "w")
#    print("SAVING", fn)
    logMsg(1,"Saving final X matrix to %s" % fn)
#    f = open(fileName, "w")
    for i in data:
        f.write(str(i[0])+"\n")
    f.close()



if __name__ == "__main__":
    print("Running")
    
    '''
    data = getLLDOSData("datasets/inside/LLS_DDOS_2.0.2-inside-all-MORE")
    f = open("inside/matrixAllMORE.txt", "w")
    for i in data:
        f.write(str(i[0])+"\n")
    f.close()
    '''    

    #print(data)
#    malPkts1, malPkts2, malPkts3 = listLLDOSLabels("phase-1-shorter-counts.txt")
#    atks = getAttacks("inside/LLS_DDOS_2.0.2-inside-phase-1", malPkts1)
#    print(atks)
#    print("1: ",malPkts1,"\n2: ",malPkts2,"\n3:",malPkts3)

    '''    
    pnts = getLLDOSLabels("datasets/inside/LLS_DDOS_2.0.2-inside-all-MORE")

    # writes attack data to file
    fi = open("testfile.txt","w") 
     
    fi.write("P1:\n")
    for i in pnts["P1"]:
        fi.write(str(i)+"\n")
    fi.write("***\nP2:\n")
    for i in pnts["P2"]:
        fi.write(str(i)+"\n")
    fi.write("***\nP3:\n")
    for i in pnts["P3"]:
        fi.write(str(i)+"\n")
    fi.close() 
    '''
    print("DONE")

