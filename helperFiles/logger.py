# Logs results to file folder
import logging

# logs a message to the previously set file
#   Input: lvl - log level (see: https://docs.python.org/3/howto/logging.html)
#          msg - string of what to log
#   Output: None
#
def logMsg(lvl, msg):
    # check if log has been set
#    if

    # check level
    if lvl == 0:
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
def setLog(fileName):
    logging.basicConfig(format='%(asctime)s  %(levelname)s: %(message)s', filename='logs/'+fileName+'.log', level=logging.DEBUG)#, filemode='w',)
    # NOTE filemode='w' overwrites previous file. 
    #      If omitted, future runs will append to file
    logging.info("Run started.")

