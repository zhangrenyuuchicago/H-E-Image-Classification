import json

msi_index = {'MSS':0, 'MSI-L':0, 'MSI-H':1}

import glob
import os
import sys
import csv

slide_msi = {}
fin = open('MSI_label.txt', 'r')
for line in fin.readlines():
    array = line.strip().split()
    slide_id = array[0]
    msi = array[1]
    slide_msi[slide_id] = msi

img_lt = glob.glob('COAD_tiles_20X_1000/COAD_tiles/*.jpg')
slide_ids = {}
for img_name in img_lt:
    basename = os.path.basename(img_name)
    array = basename.split('_')
    slide_id = array[0]

    if slide_id in slide_ids:
        slide_ids[slide_id] += 1
    else:
        slide_ids[slide_id] = 1
   
print(f'sample num: {len(slide_ids)}')

msi_num = {}
msi_slide = {}
for slide_id in slide_ids:
    pid = slide_id[:12]
    if slide_id not in slide_msi:
        continue
    msi = slide_msi[slide_id]
    if msi in msi_num:
        msi_num[msi] += 1
    else:
        msi_num[msi] = 1
    
    if msi in msi_slide:
        msi_slide[msi].append(slide_id)
    else:
        msi_slide[msi] = [slide_id]

print(f'hist of msi: {msi_num}')

import random

slide_lt = [] 
for msi in msi_slide:
    slide_lt += msi_slide[msi]

fold_slide_lt = [[] for i in range(5)]
while slide_lt:
    slide_id = slide_lt.pop()
    index = random.randint(0,4)
    fold_slide_lt[index].append(slide_id)

import json

with open('fold_slide_lt.json', 'w') as json_file:  
        json.dump(fold_slide_lt, json_file)


