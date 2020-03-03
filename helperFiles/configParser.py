import configparser
from math import floor
from fractions import Fraction
from .matrixOp import frange
from collections import OrderedDict

# NOTE File path starts where main.py executes



# TODO finish adding functionality for auto adding new configuration

config = configparser.ConfigParser()
filePath = 'config.ini'
# TODO add more to this list later
'''
listCSV = ['datasets/TrafficLabelling/Thursday-WorkingHours-Morning-SHORT-WebAttacks.pcap_ISCX.csv', 
           'datasets/inside/LLS_DDOS_2.0.2-inside-all-MORE']
dictStruct = OrderedDict([('LambdaStartValue', [str(floor((x*100))/100) for x in frange(0.01,1,0.01)]), 
                        ('Dataset', ['UNB','LLDOS']), ('CSVFile',''), 
                        ('RatioTrainData',''), ('RatioTestData',''), 
                        ('Mode',['0', '1']), ('Models',['rf','knn','svm','logreg','dtree','nb','kmeans','gb','nn']), 
                        ('LogFile','')])
'''

# writes/configures config file
# TODO UPDATE THIS
'''
def generateSection(secName):
    global config
    config[secName] = OrderedDict([('LambdaStartValue','0.01'),
                                  ('Dataset','UNB'),
                                  ('CSVFile','datasets/TrafficLabelling/Thursday-WorkingHours-Morning-8100-SHORT-WebAttacks.pcap_ISCX.csv'),
                                  ('RatioTrainData',''),
                                  ('RatioTestData',''),
                                  ('Mode','0'),
                                  ('Models','svm'),
                                  ('LogFile','trash')])
    with open(filePath, 'a') as configfile:
        config.write(configfile)
'''

# helps user create new configuration
'''
def createNewConfig():
    secName = input("\nChoose a name for the section: ")
    copyDict = dictStruct
    for key in dictStruct:
        val = None
        while not (val in dictStruct[key]):
            val = input("Give a value for %s: " % key)
        copyDict[key] = str(val)
    print("These are the values set in the section:\n", copyDict)
    exit(0)
'''

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
    return readConfig("KAGGLEV2")
#    return readConfig("MAIN")
    '''
    ready = False
    print("Available configurations:",getConfigTypes())
    print("(Type h for details on the configuration settings)")
#    print("(Type c to create a new configuration)")
    while not ready:
        con = input("\nSelect configuration (DEFAULT= enter key): ")
        if con.lower() == "h":
            printConfigDetails()
            ready = False
#        elif con.lower() == "c":
#            createNewConfig()
#            ready = False                                
        else:
            ready = True
    return readConfig(con)
    '''

# TODO add a way to change a variable for a specific run
