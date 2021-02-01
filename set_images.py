#!/usr/bin/env python

import os
import csv
import cv2
import time
import json

dir_in = './dataset/Images'
dir_outT = './dataset/train'
dir_outV = './dataset/validation'
dir_outTe = './dataset/test'
path_xl = './file/DL_modify.csv'
file_json = './file/text_mined_labels_171_and_split.json'


def read_images(dir):
    img_dirs = os.listdir(dir)
    img_dirs.sort()
    
    data = json.load(open(file_json))
    train = data.get('train_lesion_idxs')
    valid = data.get('val_lesion_idxs')
    test = data.get('test_lesion_idxs')
    
    with open(path_xl, 'rt') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            string = row[0].split(';')
            num_row = int(string[0])
            filename = string[1].replace('"', '')

            for name in img_dirs:

                if filename == name:
                    path_image = os.path.join(dir_in, name)
                    im = cv2.imread(path_image)
                    #print('read', path_image)
                    
                    for t in train:
                        if num_row == t:
                            path_out = os.path.join(dir_outT,name)
                            cv2.imwrite(path_out, im)
                            print(path_out, 'saved')
                                             
                    for v in valid:
                        if v == num_row:
                            path_out = os.path.join(dir_outV,name)
                            cv2.imwrite(path_out, im)
                            print(path_out, 'saved')
                            
                    for te in test:
                        if te == num_row:
                            path_out = os.path.join(dir_outTe,name)
                            cv2.imwrite(path_out, im)
                            print(path_out, 'saved')

    
    
if __name__ == '__main__':
    read_images(dir_in)