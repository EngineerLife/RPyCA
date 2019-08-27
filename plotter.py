# imports???
import matplotlib.pyplot as plt

def plotter(Ss,attackInds,xname="Stage 3"):
#def plotter(Ss,i,f,alpha,attackInds,myCounts,xname="Stage 3",normType=np.inf,bx=None):
#    j = 0
#    titleCnt = 0
    n = len(attackInds)
#    sNorm,normNotes = Hyp.takeNorm(Ss[i],normType=normType)
#    if bx != None: #plots what I like
#        bx.plot(sNorm,'-')
    fig, bx = plt.subplots()
    fig.subplots_adjust(left=0.2, wspace=0.6)

    bx.plot(Ss,'-')
    bx.set_ylim([0.75,1])
    if attackInds != [None]:
        bx.set_title(xname)
        bx.set_xticks(attackInds),bx.set_xticklabels(n*['|'],fontsize=6)
        bx.set_xticks(range(0,n+1,25), minor = True)
        bx.grid(which='minor', axis='x',linewidth=.5,alpha=.3)
    bx.grid(which='major', axis='both',linewidth=.5,alpha=.75)
    bx.set_xlabel("Column of Data Matrix")
    bx.set_ylabel("Value of Infinity Norm")


#    fig.align_ylabels(bx[:, 1])
    plt.show()

#    print
#    print normNotes
    #f.write("\n" + normNotes +"\n  --Epsilon: " + str(myeps[j]) +"\n")
#    FPR,TPR,TP,FP,TN,FN = Hyp.multiSet(sNorm,attackInds,alpha,myCounts)
#    ROCnotes = ("H0 alpha: " + str(alpha) +
#                " :: TP: " + str(TP) + ", FN: " + str(FN) +
#                ", TN: "+ str(TN) + ", FP: " + str(FP) + ", TPR: " + str(TPR) +
#                ", FPR: " + str(FPR) + "\nAccuracy: " + str( round(( (TP+TN)/float(TP+FN+FP+TN) ) * 100,4) ) + "%")
#    print ROCnotes
#    f.writelines("\n" + xname + "\n")
#    f.writelines(ROCnotes+"\n")
#    titleCnt += 1
#    j += 1
    print("Done")
