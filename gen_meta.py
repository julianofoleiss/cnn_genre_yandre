import glob
import os
import sys

if __name__ == '__main__':
    
    if len(sys.argv) != 3:
        print("usage: %s spec_dir meta_filename" % (sys.argv[0])) 
        exit(1)

    spec_dir = sys.argv[1]
    meta_filename = sys.argv[2]

    if spec_dir[0] == '.':
        spec_dir = spec_dir[1:]

    if spec_dir[-1] != '/':
        spec_dir += '/'

    spec_files = sorted(glob.glob("%s*.png" % (spec_dir)))

    meta_file = open(meta_filename, "w")

    for f in spec_files:
        meta_file.write("%s/%s\t%s\n" % (os.getcwd(), f, f.split("/")[1].split('_')[0]))

    meta_file.close()


    

