import configparser
from math import floor
from fractions import Fraction
from .matrixOp import frange
from collections import OrderedDict

# NOTE File path starts where main.py executes

config = configparser.ConfigParser()
filePath = 'config.ini'

# Reads config file and returns variables
def readConfig(typ='DEFAULT'):
    if typ == '':
        typ='DEFAULT'
    global config
    config.read(filePath)
    configType = config[typ]    
    return typ, OrderedDict([('LambdaStartValue',float(configType['LambdaStartValue'])),
                       ('LambdaEndValue',float(configType['LambdaEndValue'])),
                       ('LambdaIncrValue',float(configType['LambdaIncrValue'])),                       
                       ('CSVFile',configType['CSVFile']),
                       ('Labels',configType['Labels']),
                       ('OneHot',configType['OneHot']),
                       ('Skip',configType['Skip']),
                       ('RandomSeed',int(configType['randomSeed'])),
                       ('RatioTrainData',float(-1 if configType['RatioTrainData'] == '' else Fraction(configType['RatioTrainData']))),
                       ('RatioTestData',float(-1 if configType['RatioTestData'] == '' else Fraction(configType['RatioTestData']))),
                       ('Mode',int(configType['Mode'])),
                       ('Models',configType['Models']),
                       ('LogFile',configType['LogFile'])])

def printConfigDetails():
    global config
    config.read(filePath)
    for sec in getConfigTypes():
        print("TYPE:",sec)
        configType = config[sec]
        for key in configType:
            print(key,":",configType[key])
        print("------------------------------------")

def getConfigTypes():
    global config
    config.read(filePath)
    sections = config.sections()
    sections.insert(0,'DEFAULT')
    return sections

# sets the configuration variables for the run
# TODO make this actually functional later. (OR add default auto run config???)
def setConfig():
    ready = False
    print("Available configurations:",getConfigTypes())
    print("(Type h for details on the configuration settings)")
    while not ready:
        con = input("\nSelect configuration (DEFAULT= enter key): ")
        if con.lower() == "h":
            printConfigDetails()
            ready = False
        else:
            ready = True
    return readConfig(con)

