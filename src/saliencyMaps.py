import tensorflow as tf
import keras.backend as K
import numpy as np

def generateSaliency(model):

    inp = model.layers[0].input
    outp = model.layers[-1].output
    max_outp = K.max(outp, axis=1)
    saliency = tf.keras.backend.gradients(tf.keras.backend.sum(max_outp), inp)[0]
    max_class = K.argmax(outp, axis=1)
    return K.function([inp], [saliency, max_class])


def outFileName(model, x_test, outFile = None):

    # xt = x_test.reshape(-1, 402, 4)

    saliency = generateSaliency(model)([x_test, 0])

    A = [1, 0, 0, 0]
    C = [0, 1, 0, 0]
    G = [0, 0, 1, 0]
    T = [0, 0, 0, 1]

    lst_data = []
    lst_sal = []

    outputFile = open(outFile,'w')

    x_test = x_test.tolist()

    for nuc in (x_test):
        sal = saliency[0][x_test.index(nuc)]
        # import code
        # code.interact(local=dict(globals(), **locals()))
        for i in range(len(nuc)):
            if nuc[i] == A:
                lst_data.append('A')
                lst_sal.append(sal[i][A.index(1)])
            elif nuc[i] == C:
                lst_data.append('C')
                lst_sal.append(sal[i][C.index(1)])
            elif nuc[i] == G:
                lst_data.append('G')
                lst_sal.append(sal[i][G.index(1)])
            else:
                lst_data.append('T')
                lst_sal.append(sal[i][T.index(1)])
        #print(lst_sal)

        print(','.join(lst_data), file=outputFile)
        print(','.join([str(element) for element in lst_sal]), file=outputFile)

        lst_data = []
        lst_sal = []

def normalize(inputfile,outputfile):

    lines = [line.strip() for line in open(inputfile).readlines()]
    out = open(outputfile,'w')
    sm = 0.0
    count = 0
    for i in range(0,2*(len(lines)//2),2):
    
        s = [abs(float(x)) for x in lines[i+1].strip().split(',')]
        sm += sum(s)
        count += 1

    sm_avg = (sm/count)/100
    if len(lines) % 2 != 0:
        print('LEN LINES is apparently problematic!!!! (should be div by 3) --> ',len(lines))
    for i in range(0,2*(len(lines)//2),2):
        print(lines[i+0].strip(),file=out)
        s = [str(float(x)/sm_avg) for x in lines[i+1].strip().split(',')]
        print(','.join(s),file=out)
