# imports???
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA

# USE:
#    plotMat(S)
#    plotS(X, s, E, maxRank)
#    plotS(X, s, E, maxRank, True)

def plotter(Ss,attackInds,alpha,xname="Stage 3",bx=None):

    if not bx:
        print("Need plt.subplots() command for bx variable in plotter() func.")
        exit(1)
    X = range(len(Ss))
    n = len(attackInds)

#    fig, bx = plt.subplots()
#    fig.subplots_adjust(left=0.2, wspace=0.6)

    c = LA.norm(Ss,axis=1,ord=np.inf)
    # plots (x, y, type of line)
    bx.plot(c, '-b')
    bx.axhline(y=alpha, linewidth=1, color='r')
#    bx.set_ylim([0.75,1])
    
    bx.set_title(xname)
    
    bx.set_xticks(attackInds), bx.set_xticklabels(n*['|'],fontsize=6)
    bx.set_xticks(range(0,n+1,25), minor = True)
    
    bx.grid(which='minor', axis='x',linewidth=.5,alpha=.3)
    bx.grid(which='major', axis='both',linewidth=.5,alpha=.75)

    bx.set_xlabel("Column of Data Matrix")
    bx.set_ylabel("Value of Infinity Norm")


#    fig.align_ylabels(bx[:, 1])
#    plt.show('all')

    print("Done")


# plots matrices
def plotMat(mat):
    print("Plotting...")

    plt.matshow(mat)
#    plt.imshow(mat)
#    plt.colorbar()
    plt.show()


# plots Sigma matrices from PCA (SVD) and sRPCA
def plotS(T, svd, srpca, maxRank, xname, x=None, log=False):
    print("Plotting...")
    if not x:
        print("Need plt.subplots() command for x variable in plotS() func.")
        exit(1)

    T = np.asmatrix(T)
    x.plot(range(T.shape[1]), svd, 'rs', range(maxRank), srpca, 'bo')
    x.set_title(xname)

    if log:
        x.yscale("log")

#    plt.show()


###################### PROPOSAL CODE ##################################

# plotter for my proposal paper
def plotProp(mat, name, subx):
#    fig, subx = plt.subplots(1,3)

#    subx.matshow(mat)
    subx.plot(mat, 'bo')
#    subx.set_ylim([0,1])

    subx.set_title(name)

#    subx.grid(which='minor', axis='x',linewidth=.5,alpha=.3)
#    subx.grid(which='major', axis='both',linewidth=.5,alpha=.75)

    subx.set_xlabel("Features")
    subx.set_ylabel("Packets")


# NOTE NOTE NOTE this is for proposal only!!!!!
    '''
    fig = plt.figure()
    fig.subplots_adjust(left=0.07, bottom=0.21, right= 0.95, top=0.83, wspace=0.36, hspace=0.2)
    xMat = fig.add_subplot(1, 3, 1)
    lMat = fig.add_subplot(1, 3, 2)
    sMat = fig.add_subplot(1, 3, 3)
    plotProp(X, "X Matrix", xMat)
    plotProp(L, "L Matrix", lMat)
    plotProp(S, "S Matrix", sMat)

    plt.show()
    exit(0)
    '''


