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
    out.write("exp;acc;precision;recall;f1;avg_fold_train_time;final_val_err;best_epoch;last_epoch;\n")

    for ipt in ipts:

        with codecs.open(ipt, encoding='utf-8') as f:
            c = f.readlines()

        accs = []
        perf = []
        times = []
        val_err = []
        max_epochs = 0
        best_epoch = []
        last_epoch = []

        for i in xrange(len(c)):

            if "num_epochs" in c[i]:
                d = c[i].split(" ")
                max_epochs = int(d[-1])

            if "Epoch 1 " in c[i]:
                fold_times = []
            
            if "took" in c[i]:
                t = c[i].split(" ")[-1]
                t = t[:-2]
                fold_times.append(float(t))

            if "Accuracy: " in c[i]:
                d = c[i].split(" ")
                accs.append(float(d[1].strip()))
        
            if "validation error:" in c[i]:
                d = c[i].split(" ")
                if d[0] == "STOPPED":
                    val_err.append(float(d[8]))
                    le = int(d[4][:-1])
                    be = int(d[10][:-2])
                    best_epoch.append(be)
                    last_epoch.append(le)

                if d[0] == "EXECUTED":
                    val_err.append(float(d[6]))
                    best_epoch.append(max_epochs)
                    last_epoch.append(max_epochs)

            if "avg / total" in c[i]:
                d = filter(None, c[i].split(" "))
                prec = float(d[3])
                rec = float(d[4])
                f1 = float(d[5])
                supp = int(d[6].strip())

                perf.append( [prec, rec, f1, supp] )
                times.append(np.sum(fold_times))

        #print val_err

        m = np.array(perf, dtype=object)

        f = m[:,:-1]

        f_avg = np.round(np.mean(f, axis=0, dtype='float64'), decimals=2)
        f_std = np.round(np.std(f, axis=0, dtype='float64'), decimals=2)

        acc_avg = np.round(np.mean(accs, axis=0, dtype='float64'), decimals=2)
        acc_std = np.round(np.std(accs, axis=0, dtype='float64'), decimals=2)

        time_avg = np.round(np.mean(times, dtype='float64'), decimals=2)
        time_std = np.round(np.std(times, dtype='float64'), decimals=2)

        err_avg = np.round(np.mean(val_err, dtype='float64'), decimals=2)
        err_std = np.round(np.std(val_err, dtype='float64'), decimals=2)

        be_avg = np.round(np.mean(best_epoch, dtype='float64'), decimals=2)
        be_std = np.round(np.std(best_epoch, dtype='float64'), decimals=2)

        le_avg = np.round(np.mean(last_epoch, dtype='float64'), decimals=2)
        le_std = np.round(np.std(last_epoch, dtype='float64'), decimals=2)

        print acc_avg, acc_std, f_avg, f_std, time_avg, time_std, err_avg, err_std, be_avg, be_std, le_avg, le_std

        out.write("%s;%.2f +- %.2f; %.2f +- %.2f; %.2f +- %.2f; %.2f +- %.2f; %.2f +- %.2f; %.2f +- %.2f; %.2f +- %.2f; %.2f +- %.2f;\n" % 
            (ipt, acc_avg, acc_std, f_avg[0], f_std[0], f_avg[1], f_std[1], f_avg[2], f_std[2], time_avg, time_std, err_avg, err_std, be_avg, be_std, le_avg, le_std))

    out.close()

