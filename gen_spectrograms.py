import subprocess
from multiprocess import Pool
import glob
import os
import sys

def output_spectrogram(path):
    command = "sox %s -n channels 1 spectrogram -y 256 -z %d -r -o %s %s" % (path[0], path[2], path[1], path[3])
    print ("executing %s" % command)
    subprocess.call(command, shell=True)


if __name__ == "__main__":
    
    if len(sys.argv) < 5:
        print ("usage: %s input_dir output_dir audio_extension z [grayscale]" % sys.argv[0])
        exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    extension = sys.argv[3]
    z = int(sys.argv[4])
    grayscale = False

    if 'grayscale' in sys.argv:
        grayscale = True

    if input_dir[-1] != '/':
        input_dir += '/'

    if output_dir[-1] != '/':
        output_dir += '/'

    print "%s*.%s" % (input_dir, extension)

    audios = sorted(glob.glob("%s*.%s" % (input_dir, extension)))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if len(audios) <= 0:
        print("No audio files found in input directory...")
        exit(1)

    output_files = []

    for i in audios:
        d = i.split("/")[-1]
        filename = d.split(".")[0]
        output_files.append( "%s%s.png" % (output_dir, filename) )

    zs = [z] * len(output_files)
    gs = ['-m' if grayscale else ''] * len(output_files)
    work = zip(audios, output_files, zs, gs)

    pool = Pool(4)

    pool.map(output_spectrogram, work)
