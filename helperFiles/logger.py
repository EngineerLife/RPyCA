# Logs results to file folder
import logging

logSET = False
overrideLevel = False   # TODO incorporate variable???

# logs a message to the previously set file
#   Input: lvl - log level (see: https://docs.python.org/3/howto/logging.html)
#          msg - string of what to log
#   Output: None
#
def logMsg(lvl, msg):
    # check if log has been set
    global logSET
    if not logSET:
        return

    # check level
    if lvl == -1 or overrideLevel:
        # logger should not and does not log event
        return
    elif lvl == 0:
        logging.debug(msg)
    elif lvl == 1:
        logging.info(msg)
    elif lvl == 2:
        logging.warning(msg)
    elif lvl == 3:
        logging.error(msg)
    elif lvl == 4:
        logging.critical(msg)
    else:
        print("Level out of bounds. Nothing logged")


# starts log with a filename
#   Input: fileName - string name of the file to log to
#   Output: None
def setLog(fileName, override=False):
    logging.basicConfig(format='%(asctime)s  %(levelname)s: %(message)s', filename='logs/'+fileName+'.log', level=logging.DEBUG)#, filemode='w',)
    # NOTE filemode='w' overwrites previous file. 
    #      If omitted, future runs will append to file
    logging.info("Run started.")
    if override:
        logging.info("Logging has been overridden.")
        global overrideLevel
        overrideLevel = True
    global logSET
    logSET = True

# Manually override logs with this function
def overrideLogs():
    logging.info("Logging has been overridden.")
    global overrideLevel
    overrideLevel = True

# Manually stop overriding logs with this function
def continueLogs():
    logging.info("Logging has resumed.")
    global overrideLevel
    overrideLevel = False

