import subprocess
import glob
import sys
import os
from multiprocessing import Pool
import traceback

def shellquote(s):
    return "'" + s.replace("'", "'\\''") + "'"

def get_song_duration(filename):
    #filename = shellquote(filename)
    command = ['soxi', filename]

    proc = subprocess.check_output(command)
    proc = proc.replace(" ", "")
    proc = proc.split("\n")

    duration = proc[5].split(":")

    duration = float(duration[1]) * 3600 + float(duration[2]) * 60 + float(duration[3].split("=")[0])
    
    return duration

def human_readable(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s

def trim_middle(infile, outfile, duration, length, mix_channels):
    infile = shellquote(infile)
    outfile = shellquote(outfile)

    if (30 + length) > duration:
        ini = (duration / 2) - (length / 2)
        end = ini + length
    else:
        if duration < length:
            ini = 0
            end = length
        else: 
            ini = 30
            end = ini + length 

    ini = int(ini)
    end = int(end)

    print ini, end, (end - ini)

    h, m, s = human_readable(ini)
    ini = "%02d:%02d:%02d" % (h, m, s)

    h, m, s = human_readable(end)
    end = "%02d:%02d:%02d" % (h, m, s)

    command = "sox %s %s trim %s =%s" % (infile, outfile, str(ini), str(end))
    #print command
    if mix_channels:
        command += " channels 1 "

    subprocess.call(command, shell=True)

def trimmer_thread(work):
    try:
        song = work[0]
        path = work[1]
        duration = work[2]
        length = work[3]
        mix_channels = work[4]

        print ("trimming %s (%.2fs) to %ds... %s" %  (path, duration, length, "mixing channels" if mix_channels else "")  )

        trim_middle(song, path, duration, length, mix_channels)

    except Exception:
        traceback.print_exc()
        raise

if __name__ == "__main__":

    if len(sys.argv) < 5:
        print("usage: %s song_folder clipped_folder clipped_size extension [mixchannels]" % (sys.argv[0]))
        exit(1)

    mp3folder = sys.argv[1]
    clipped_folder = sys.argv[2]
    clipped_size = int(sys.argv[3])
    extension = sys.argv[4]

    mix_channels = True if "mixchannels" in sys.argv else False

    if clipped_folder[-1] != "/":
        clipped_folder +="/"

    if not os.path.exists(clipped_folder):
        os.makedirs(clipped_folder)

    songs = sorted([i for i in glob.glob( mp3folder + '/*.' + extension)])

    pool = Pool(2)

    work = []

    for song in songs:
        duration = get_song_duration(song)
        filename = os.path.splitext(song)[0].split("/")[-1]
        work.append((song, clipped_folder + filename + "." + extension, duration, clipped_size, mix_channels))

    pool.map(trimmer_thread, work)
    
