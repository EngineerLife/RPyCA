import sys, configparser
from math import floor
from fractions import Fraction
from .matrixOp import frange
from collections import OrderedDict

# NOTE File path starts where main.py executes

config = configparser.ConfigParser()
filePath = 'config.ini'

# Reads config file and returns variables
# TODO enforce input types here
def readConfig(typ='DEFAULT'):
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
                       ('SampleSize',float(configType['SampleSize'])),
                       ('RatioTrainData',float(-1 if configType['RatioTrainData'] == '' else Fraction(configType['RatioTrainData']))),
                       ('RatioValidData',float(-1 if configType['RatioValidData'] == '' else Fraction(configType['RatioValidData']))),
                       ('Mode',int(configType['Mode'])),
                       ('Models',configType['Models']),
                       ('LogFile',configType['LogFile'])])

# sets the configuration variables for the run
# Input: User input string of the configuration name
# Output: OrderedDict of config variables
def setConfig():
    return readConfig(sys.argv[1])
