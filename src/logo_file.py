
# This python script processes the contribution scores for creating sequence logos.

import numpy as np
import code

# This function calculates average of scores for all nucleotides at a certain position.
# It takes in two lists for sequences and scores, and a position.

def Score(nucleotidesList, scoresList, pos):
    scores = {'A': [], 'C': [], 'G': [], 'T': []}
    for nucs,scrs in zip(nucleotidesList,scoresList):
        n = nucs[pos]   #single nucleotide
        s = scrs[pos]

        scores[n].append(s)

    for k,v in scores.items():
        if scores[k]:
            scores[k] = np.mean(v)
        else:
            scores[k] = 0.0

    return scores


# This function writes the processed contribution scores in pandas dataframe format for logomaker.

def LogoFile(input,output):
    import pandas as pd

    allLines = open(input).readlines()
    #nucleotides in a sequence
    nuc = [allLines[i].rstrip().split(',') for i in range(0,len(allLines),2)]
    #scores in a sequence
    score = [[float(a) for a in allLines[i].rstrip().split(',')] for i in range(1,len(allLines),2)]

    '''#only use the TP samples
    nucTP = []
    scrTP =[]
    for i in range(0,len(m)):
        if m[i][0]>=0.5 and m[i][1]==1:
            nucTP.append(nuc[i])
            scrTP.append(score[i])'''

    files = '{}_{}.txt'.format(output,'TP')

    #write the scores into a dictionary then convert it into dataframe.
    scoresTP = {'A': [], 'C': [], 'G': [], 'T': []}
    for pos in range(0,len(nuc[0])):
        s = Score(nuc,score,pos)
        scoresTP['A'].append(s['A'])
        scoresTP['C'].append(s['C'])
        scoresTP['G'].append(s['G'])
        scoresTP['T'].append(s['T'])
    TP = pd.DataFrame(scoresTP)
    TP.to_pickle(files)