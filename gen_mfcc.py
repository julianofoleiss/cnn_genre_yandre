import mir3.modules.features as feat
import mir3.modules.tool.wav2spectrogram as spec
import mir3.modules.features.mfcc as mfcc
import mir3.modules.features.join as join
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import os
from multiprocess import Pool

def calc_mfcc(filename, mfcc_coeff, fs):

    converter = spec.Wav2Spectrogram()
    s = converter.convert(open(filename), window_length=2048, window_step=1024, spectrum_type='magnitude', save_metadata=True, fs=fs)
    mfcc_ex = mfcc.Mfcc()

    track = mfcc_ex.calc_track(s, mfcc_coeff)

    return track.data

def thread_mfcc(work):
    filename = work[0]
    output_file = work[1]
    coeffs = work[2]
    fs = work[3]

    print("Calculating MFCC coefficients for %s" % filename)

    mfccs = calc_mfcc(filename, coeffs, fs)
    
    mfccs = 20 * np.log( np.power(mfccs, 2) + np.finfo(float).eps)
    mfccs = mfccs.T
    mfccs = mfccs[:,0 : (mfccs.shape[1] / 16) * 16 ]

    plt.imsave(output_file, mfccs[::-1], cmap=plt.get_cmap('Greys'))

    return mfccs.shape[1]

if __name__ == '__main__':

    if len(sys.argv) < 6:
        print ("usage: %s input_dir output_dir audio_extension mfcc_coeffs sampling_rate" % sys.argv[0])
        exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    extension = sys.argv[3]
    mfcc_coeffs = int(sys.argv[4])
    fs = int(sys.argv[5])
    
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

    cs = [mfcc_coeffs] * len(audios)
    fss = [fs] * len(audios)
    work = zip(audios, output_files, cs, fss)

    print work[0]

    thread_mfcc(work[0])

    pool = Pool(4)

    r = pool.map(thread_mfcc, work)

    print("Average resulting width: %d" % np.mean(r))



