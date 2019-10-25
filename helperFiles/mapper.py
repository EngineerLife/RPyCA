import math
import numpy as np

# mapping values function
def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

# maps the stuff
def mapper(T, S):
    # map values to 0-1
    newS, newSS = [], []#np.matrix()
    leftMax = np.amax(S)
    for row in range(T.shape[0]):
        for i in range(T.shape[1]):
    #            print(S[row,i])
            newS.append(translate(S[row,i], 0, leftMax, 0, 1))
        newSS.append(newS)
        newS = []
    return np.matrix(newSS)

# USE:
#    X = mapper(T, np.matrix(X))
#    S = mapper(T, S)
#    print("Mapped Dense S: ", S)  
