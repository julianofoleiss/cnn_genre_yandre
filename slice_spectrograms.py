import os
import sys
import glob
from PIL import Image
from multiprocess import Pool

def slice_track(spec_img, output_folder, slice_width, slice_height):

    full_filename = spec_img.split("/")[-1]
    filename = full_filename.split(".")[0]
    
    print("slicing %s to %s in %d x %d patches" % (spec_img, output_folder, slice_height, slice_width))

    sliced = "%s/%s" % (output_folder, filename)
    sliced = sliced + "_%02d.png"

    img = Image.open(open(spec_img))

    k = 0
    for x in xrange(0, img.size[0], slice_width):
        cropped = img.crop((x, 0, x + slice_width, slice_height))
        cropped.save(sliced % (k))
        k+=1

def slice_thread(args):
    spec_img = args[0]
    outf = args[1]
    slicew = args[2]
    sliceh = args[3]

    slice_track(spec_img, outf, slicew, sliceh)


if __name__ == "__main__":

    if len(sys.argv) < 5:
        print("usage: %s spec_folder output_folder slice_width slice_height" % (sys.argv[0]))
        exit(1)

    spec_folder = sys.argv[1]
    output_folder = sys.argv[2]
    slice_width = int(sys.argv[3])
    slice_height = int(sys.argv[4])

    specs = sorted(glob.glob("%s/*.png" % (spec_folder)))

    if len(specs) == 0:
        print("Could not find any PNG spectrograms in %s" % (output_folder))
        exit(1)

    if output_folder[-1] != "/":
        output_folder +="/"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    slice_track(specs[0], output_folder, 16, 256)

    slicew = [slice_width] * len(specs)
    sliceh = [slice_height] * len(specs)
    outf = [output_folder] * len(specs)

    work = zip(specs, outf, slicew, sliceh)

    pool = Pool(4)

    pool.map(slice_thread, work)

