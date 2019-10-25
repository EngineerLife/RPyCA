# TODO create function for this

# Computes TP, FP, TN, FN
    '''
    TP, FP, TN, FN = 0, 0, 0, 0
    for row in range(T.shape[0]):
        for feat in range(T.shape[1]):
            # not an attack
            if S[row, feat] < alpha:
                if row in malPkts:  # an attack, False Negative
                    FN += 1
                else:   # not an attack, True Negative
                    TN += 1
            else:   # an attack
                if row in malPkts:  # an attack, True Positive
                    TP += 1
                else:   # not an attack, False Positive
                    FP += 1
    # Compute the False Positive and True Positive Rates
    FPR = FP*1./(FP + TN)
    TPR = TP*1./(TP + FN)

    print("Lambda: ", l, "  Alpha: ", alpha)
    print("\nFPR:", FPR, "    TPR:", TPR)
#    print("# of attacks: ", len(malPkts))
