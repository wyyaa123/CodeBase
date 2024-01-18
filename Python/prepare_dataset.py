# -*- encoding: utf-8 -*-
'''
@File    :   prepare_dataset.py
@Time    :   2024/01/07 21:45:35
@Author  :   orCate 
@Version :   1.0
@Contact :   8631143542@qq.com
'''

# here put the import lib

import os
import re
import shutil

# create input directory and target directory 
os.makedirs("input", exist_ok=True)
os.makedirs("target", exist_ok=True)

script_source_dir = os.path.dirname(os.path.abspath(__file__))

test_dirs = [os.path.join(script_source_dir, "val_blur"), os.path.join(script_source_dir, "val_sharp")]
names = ["input", "target"]

pattern = re.compile(r'/0+(\d+)')
for name, test_dir in zip(names, test_dirs):
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            abs_path = os.path.join(root, file)
            relative_path = os.path.relpath(abs_path, test_dir)
            relative_path = pattern.sub(lambda x: '/' + x.group(1), relative_path)
            shutil.copy2(abs_path, os.path.join(name, relative_path.replace("/", "_")))
            # cv.imread(abs_path, cv.IMREAD_COLOR)
            # print (relative_path)
            # print (abs_path)
        

# print (script_source_dir)
