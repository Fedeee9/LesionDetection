#!/usr/bin/env python

import numpy as np
import os
import cv2
import csv

dir_in = './dataset/Images_png'
dir_out = './dataset/Images'
out_fmt = './dataset/%s_%03d.png'  # format of the file name to output
info_fn = './file/DL_info.csv'  # file name of the information file


def load_slices(dir, slice_idxs, fn_out):
    """load slices from 16-bit png files"""
    slice_idxs = np.array(slice_idxs)
    assert np.all(slice_idxs[1:] - slice_idxs[:-1] == 1)
    ims = []
    i=0
    for slice_idx in slice_idxs:
        fn = '%03d.png' % slice_idx
        fn2 = out_fmt % (dir, slice_idxs[i])
        path = os.path.join(dir_in, dir, fn)
        im = cv2.imread(path, -1)  # -1 is needed for 16-bit image
        assert im is not None, 'error reading %s' % path
        print ('read', path)
        
        img = (im.astype(np.int32) - 32768).astype(np.int16)

        path_out = os.path.join(dir_out, fn2)
        cv2.imwrite(path_out, img)
        print (fn2, 'saved')
        i = i+1


def read_DL_info():
    """read spacings and image indices in DeepLesion"""
    spacings = []
    idxs = []
    with open(info_fn, 'rt') as csvfile:
        reader = csv.reader(csvfile)
        rownum = 0
        for row in reader:
            if rownum == 0:
                header = row
                rownum += 1
            else:
                idxs.append([int(d) for d in row[1:4]])
                spacings.append([float(d) for d in row[12].split(',')])

    idxs = np.array(idxs)
    spacings = np.array(spacings)
    return idxs, spacings


if __name__ == '__main__':
    idxs, spacings = read_DL_info()
    if not os.path.exists(dir_out):
        os.mkdir(dir_out)
    img_dirs = os.listdir(dir_in)
    img_dirs.sort()
    
    for dir1 in img_dirs:
        # find the image info according to the folder's name
        idxs1 = np.array([int(d) for d in dir1.split('_')])
        i1 = np.where(np.all(idxs == idxs1, axis=1))[0]
        spacings1 = spacings[i1[0]]

        fns = os.listdir(os.path.join(dir_in, dir1))
        slices = [int(d[:-4]) for d in fns if d.endswith('.png')]
        slices.sort()

        # Each folder contains png slices from one series (volume)
        # There may be several sub-volumes in each volume depending on the key slices
        # We group the slices into sub-volumes according to continuity of the slice indices
        groups = []
        for slice_idx in slices:
            if len(groups) != 0 and slice_idx == groups[-1][-1]+1:
                groups[-1].append(slice_idx)
            else:
                groups.append([slice_idx])
        for group in groups:
            # group contains slices indices of a sub-volume
            fn_out = out_fmt % (dir1, group[0])
            load_slices(dir1, group, fn_out)