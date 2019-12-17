import configparser
import constraint
from collections import OrderedDict

# NOTE File path starts where main.py executes

config = configparser.ConfigParser()
filePath = 'config.ini'
dictStruct = OrderedDict([('LambdaStartValue',''), ('Dataset',''), ('CSVFile',''), 
            ('RatioTrainData',''), ('RatioTestData',''), ('Mode',''), ('Models',''), ('LogFile','')])

# writes/configures config file
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

# helps user create new configuration
def createNewConfig():
    secName = input("\nChoose a name for the section: ")
    copyDict = dictStruct
    for key in dictStruct:
        val = input("Give a value for %s: " % key)

        copyDict[key] = str(val)
    print("These are the values set in the section:\n", copyDict)

# Reads config file and returns variables
def readConfig(typ='DEFAULT'):
    if typ == '':
        typ='DEFAULT'
    global config
    config.read(filePath)
    configType = config[typ]    
    return OrderedDict([('LambdaStartValue',float(configType['LambdaStartValue'])),
                       ('Dataset',configType['Dataset']),
                       ('CSVFile',configType['CSVFile']),
                       ('RatioTrainData',float(-1 if configType['RatioTrainData'] == '' else configType['RatioTrainData'])),
                       ('RatioTestData',float(-1 if configType['RatioTestData'] == '' else configType['RatioTestData'])),
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
def setConfig():
    ready = False
    print("Available configurations:",getConfigTypes())
    print("(Type h for details on the configuration settings)")
    print("(Type c to create a new configuration)")
    while not ready:
        con = input("\nSelect configuration (DEFAULT= enter key): ")
        if con.lower() == "h":
            printConfigDetails()
            ready = False
        elif con.lower() == "c":
            createNewConfig()
            ready = False                                
        else:
            ready = True
    return readConfig(con)

# TODO add a way to change a variable for a specific run
