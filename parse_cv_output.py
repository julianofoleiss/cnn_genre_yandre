import codecs
import sys
import numpy as np

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print ("Usage: %s cv_experiment_output [more_outputs] output.csv")
        exit(1)

    ipts = []

    for i in xrange(1, len(sys.argv)-1):
        ipts.append(sys.argv[i])

    opt = sys.argv[-1]

    out = codecs.open(opt, mode='w', encoding='utf-8')
    out.write("exp;acc;precision;recall;f1;\n")

    for ipt in ipts:

        with codecs.open(ipt, encoding='utf-8') as f:
            c = f.readlines()

        accs = []
        perf = []

        for i in xrange(len(c)):

            if "Accuracy: " in c[i]:
                d = c[i].split(" ")
                accs.append(float(d[1].strip()))
        
            if "avg / total" in c[i]:
                d = filter(None, c[i].split(" "))
                prec = float(d[3])
                rec = float(d[4])
                f1 = float(d[5])
                supp = int(d[6].strip())

                perf.append( [prec, rec, f1, supp] )


        m = np.array(perf, dtype=object)

        f = m[:,:-1]

        f_avg = np.round(np.mean(f, axis=0, dtype='float64'), decimals=2)
        f_std = np.round(np.std(f, axis=0, dtype='float64'), decimals=2)

        acc_avg = np.round(np.mean(accs, axis=0, dtype='float64'), decimals=2)
        acc_std = np.round(np.std(accs, axis=0, dtype='float64'), decimals=2)

        print acc_avg, acc_std, f_avg, f_std

        out.write("%s;%.2f +- %.2f; %.2f +- %.2f; %.2f +- %.2f; %.2f +- %.2f;\n" % (ipt, acc_avg, acc_std, f_avg[0], f_std[0], f_avg[1], f_std[1], f_avg[2], f_std[2]))

    out.close()

