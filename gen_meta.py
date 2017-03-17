import glob
import os
import sys
import codecs

if __name__ == '__main__':
    
    if len(sys.argv) < 3:
        print("usage: %s spec_dir meta_filename [label_file is_slices]" % (sys.argv[0])) 
        exit(1)

    spec_dir = sys.argv[1]
    meta_filename = sys.argv[2]

    label_file = None
    is_slices = False
    
    if len(sys.argv) > 3:
        label_file = sys.argv[3]
    
    if len(sys.argv) > 4:
        is_slices = True if sys.argv[4] in ["y", "1", "True", "true"] else False

    if spec_dir[0] == '.':
        spec_dir = spec_dir[1:]

    if spec_dir[-1] != '/':
        spec_dir += '/'

    spec_files = sorted(glob.glob("%s*.png" % (spec_dir)))

    meta_file = open(meta_filename, "w")

    if label_file == None:
        for f in spec_files:
            meta_file.write("%s/%s\t%s\n" % (os.getcwd(), f, f.split("/")[1].split('_')[0]))
           
    else:
        labels = dict()

        with codecs.open(label_file, encoding='utf-8') as f:
            c = f.readlines()
        
        for l in c:
            d = l.split("\t")
            labels[ os.path.basename(d[0]).split(".")[0] ] = d[1].strip()

        for f in spec_files:
            k = os.path.basename(f).split(".")[0] if not is_slices else os.path.basename(f).split(".")[0][:-3]
            meta_file.write("%s/%s\t%s\n" % (os.getcwd(), f, labels[k]))

    meta_file.close()


    

