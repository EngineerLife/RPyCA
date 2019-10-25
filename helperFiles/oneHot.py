import math, sys
import numpy as np
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# Function one hot encodes data
#   takes in column of data to encode
#   returns numpy matrix of encoded column data
def oneHot(clmn):
    # define example
    values = clmn
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # binary encode
    onehot_encoder = OneHotEncoder(categories='auto', sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # invert first example
    inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])

    return onehot_encoded

# makes a list of features 
# useful for one-hot as # clmns vary
def makeFeat(lis, num, featName):
    for i in range(num):
        lis.append(featName)
    return lis


######################################  X MATRIX CREATION FUNCTIONS ######################################

# ONLY USED FOR THE PROPOSAL/OG KATHLEEN PAPER
# PROPOSAL ONLY!!!!!!
# EXAMPLE: ['172.16.112.50' '172.16.113.148' 'TELNET'   '60'      '23'        '4170']
#               src ip          dest ip      protocol   len    src port     dest port
def createMatrixProposal(X):
    # Source IP
    sip = X[:,0].T
    siph = []
    # NOTE only uses the first byte of IP address
    for r in range(X.shape[0]):
        adr = sip[0,r]
        newAdr = adr.split(".")[0]
        siph.append(newAdr)
    sip = np.array(siph)#, dtype=float)
    sipOH = oneHot(sip)

    # Destination IP
    dip = X[:,1].T
    diph = []
    # NOTE only uses the first byte of IP address
    for r in range(X.shape[0]):
        adr = dip[0,r]
        newAdr = adr.split(".")[0]
        diph.append(newAdr)
    dip = np.array(diph)#, dtype=float)
    dipOH = oneHot(dip)

    # Protocols
    p = X[:,2].T
    ph = []
    for r in range(X.shape[0]):
        ph.append(p[0,r])
    p = np.array(ph)#, dtype=float)
    pOH = oneHot(p)

    # Packet Length
    plm = X[:,3]
    plmh = []
    for r in range(X.shape[0]):
        plmh.append(float(plm[r,0]))
    plm = np.array(plmh, dtype=float)
    plmd = plm.shape[0]
    plm = np.reshape(plm, (plmd,1))


    # Source Port
    sp = X[:,4].T
    sph = []
    spm = []
    for r in range(X.shape[0]):
        if sp[0,r] is None:     # handle missing port numbers
            spm.append(1)   # designate missing port
            sph.append(-1)
        else:
            spm.append(0)   # designate port available
            sph.append(sp[0,r])
    spm = np.array(spm, dtype=int)
    spmd = spm.shape[0]
    spm = np.reshape(spm, (spmd,1))
    sp = np.array(sph, dtype=int)
    sp[sp > 1024] = 1024
    spOH = oneHot(sp)

    # Destination Port
    dp = X[:,5].T
    dph = []
    dpm = []
    for r in range(X.shape[0]):
        if dp[0,r] is None:     # handle missing port numbers
            dpm.append(1)   # designate missing port
            dph.append(-1)
        else:
            dpm.append(0)   # designate port available
            dph.append(dp[0,r])
    dpm = np.array(dpm, dtype=int)
    dpmd = dpm.shape[0]
    dpm = np.reshape(dpm, (dpmd,1))
    dp = np.array(dph, dtype=int)
    dp[dp > 1024] = 1024
    dpOH = oneHot(dp)

    # Distingishing destination ports
    dp[dp < 1024] = 0
    dp[dp >= 1024] = 1
    ddpOH = oneHot(dp)

    # Distinguishing source ports
    sp[sp < 1024] = 0
    sp[sp >= 1024] = 1
    dspOH = oneHot(sp)

    # creates new X matrix
#    print(sipOH.shape, spOH.shape, dipOH.shape, dpOH.shape, pOH.shape, dspOH.shape, ddpOH.shape, plm.shape, spm.shape, dpm.shape)
    newX = np.concatenate((sipOH, dipOH, spOH, dpOH, dspOH, ddpOH, spm, dpm, pOH, plm), axis=1)

    # MUST UPDATE with *NEW* FEATURES (not change in one-hot amt of clmns)
    feats = []
    feats = makeFeat(feats, sipOH.shape[1], "Src IP")
    feats = makeFeat(feats, dipOH.shape[1], "Dest IP")
    feats = makeFeat(feats, spOH.shape[1], "Src Port")
    feats = makeFeat(feats, dpOH.shape[1], "Dest Port")
    feats = makeFeat(feats, dspOH.shape[1], "Dist Src Port")
    feats = makeFeat(feats, ddpOH.shape[1], "Dist Dest Port")
    feats = makeFeat(feats, spm.shape[1], "Missing Src Port")
    feats = makeFeat(feats, dpm.shape[1], "Missing Dest Port")
    feats = makeFeat(feats, pOH.shape[1], "Protocol")
    feats = makeFeat(feats, plm.shape[1], "Packet Length")
#    print(len(feats))  # prints number of features used

    return newX



# ONLY USED FOR THE UNB DATASET MAIN THESIS
def createMatrix(X):
    # Source IP
    sip = X[:,0].T
    siph = []
    # NOTE only uses the first byte of IP address
    for r in range(X.shape[0]):
        adr = sip[0,r]
        newAdr = adr.split(".")[0]
        siph.append(newAdr)
    sip = np.array(siph, dtype=int)
    sipOH = oneHot(sip)

    # Source Port
    sp = X[:,1].T
    sph = []
    for r in range(X.shape[0]):
        sph.append(sp[0,r])
    sp = np.array(sph, dtype=int)
    sp[sp > 1024] = 1024
    spOH = oneHot(sp)

    # Destination IP
    dip = X[:,2].T
    diph = []
    # NOTE only uses the first byte of IP address
    for r in range(X.shape[0]):
        adr = dip[0,r]
        newAdr = adr.split(".")[0]
        diph.append(newAdr)
    dip = np.array(diph, dtype=int)
    dipOH = oneHot(dip)

    # Destination Port
    dp = X[:,3].T
    dph = []
    for r in range(X.shape[0]):
        dph.append(dp[0,r])
    dp = np.array(dph, dtype=int)
    dp[dp > 1024] = 1024
    dpOH = oneHot(dp)

    # Protocols
    p = X[:,4].T
    ph = []
    for r in range(X.shape[0]):
        ph.append(p[0,r])
    p = np.array(ph, dtype=int)
    pOH = oneHot(p)

    # Packet Length Mean
    plm = X[:,5]
    plmh = []
    for r in range(X.shape[0]):
        plmh.append(float(plm[r,0]))
    plm = np.array(plmh, dtype=int)
    plmd = plm.shape[0]
    plm = np.reshape(plm, (plmd,1))

    # TODO do we care about keeping these?????
    # Distingishing destination ports
    dp[dp < 1024] = 0
    dp[dp >= 1024] = 1
    ddpOH = oneHot(dp)

    # Distinguishing source ports
    sp[sp < 1024] = 0
    sp[sp >= 1024] = 1
    dspOH = oneHot(sp)

    # creates new X matrix
#    print(sipOH.shape, spOH.shape, dipOH.shape, dpOH.shape, pOH.shape, dspOH.shape, ddpOH.shape, plm.shape, spm.shape, dpm.shape)
    newX = np.concatenate((sipOH, dipOH, spOH, dpOH, dspOH, ddpOH, pOH, plm), axis=1)

    # MUST UPDATE with *NEW* FEATURES (not change in one-hot amt of clmns)
    feats = []
    feats = makeFeat(feats, sipOH.shape[1], "Src IP")
    feats = makeFeat(feats, dipOH.shape[1], "Dest IP")
    feats = makeFeat(feats, spOH.shape[1], "Src Port")
    feats = makeFeat(feats, dpOH.shape[1], "Dest Port")
    feats = makeFeat(feats, dspOH.shape[1], "Dist Src Port")
    feats = makeFeat(feats, ddpOH.shape[1], "Dist Dest Port")
    feats = makeFeat(feats, pOH.shape[1], "Protocol")
    feats = makeFeat(feats, plm.shape[1], "Packet Length")
#    print(len(feats))  # prints number of features used

    return newX
