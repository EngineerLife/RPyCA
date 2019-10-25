from collections import Counter

# prints stats on file
def printStats(filename):
    c, mal, m, a, ogMal = 0, 0, 0, 0, 0
    sourceIps, malSourceIps, malTime = [], [], []

    with open(filename) as fin:
        for line in fin:
            lineInfo = line.split(",")
            # label
            if not "BENIGN" in lineInfo[84]:
                if not lineInfo[1] in malSourceIps:
                    malSourceIps.append(lineInfo[1])
                mal += 1

            # source IP
            if not lineInfo[1] in sourceIps:
                sourceIps.append(lineInfo[1])

            # timestamp
            if c == 0:  # skips header row
                c += 1
                ogMal = mal
                continue
            time = lineInfo[6].split(" ")[1]
            hour = int(time[:2].replace(":", ""))
            if hour >= 8 and hour < 12:
                timeFrame1 = "Morning"
                m += 1
                if mal > ogMal and not timeFrame1 in malTime:
                    malTime.append(timeFrame1)
            elif hour >= 1 and hour <= 5:
                timeFrame2 = "Afternoon"
                a += 1
                if mal > ogMal and not timeFrame2 in malTime:
                    malTime.append(timeFrame2)

            c += 1
            ogMal = mal

    sortedSource = sorted(sourceIps)
    dictSource = dict(Counter(sortedSource))
    sortedMalSource = sorted(malSourceIps)
    dictMalSource = dict(Counter(sortedMalSource))

    # All totals subtract the label line
    print("Total # packets:", c-1)
    print("Total # malicious packets:", mal-1)
    print("Percent of malicious to total", ((mal-1)/(c-1)*100))
    print("# Unique Source IP's:", len(dictSource)-1)
    print("# Unique Malicious Source IP's:", len(dictMalSource)-1)
    print("Malicious packets during:", malTime)
    print("# packets during", timeFrame1, "=", m)
    print("# packets during", timeFrame2, "=", a)
