#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 09:30:38 2020

@author: yuanhanglin
"""

import pandas as pd
import numpy as np

mapping_method = ["no-mapping", "linear-transformation", 
                  "affine-transformation", "decision-tree", "random-forest"]

dates = ["130411", "130606", "131125",
         "140416", "140600", "140829",
         "150416", "150602", "150824"]

date2idx = {date : idx for idx, date in enumerate(dates)}

date2season = {} 
date2season["130411"] = "Apr2013"
date2season["130606"] = "Jun2013"
date2season["131125"] = "Nov2013"
date2season["140416"] = "Apr2014"
date2season["140600"] = "Jun2014"
date2season["140829"] = "Aug2014"
date2season["150416"] = "Apr2015"
date2season["150602"] = "Jun2015"
date2season["150824"] = "Nov2015"

result_tables = {}
for key in mapping_method:
    result_tables[key] = [[[] for _ in range(9)] for _ in range(9)]
    
file_name = "PUBLISH_RESULT.txt"

with open(file_name) as file:
    for line in file:
        if line[0].isdigit():
            _, train_date, _, test_date, accuracy, method = line.split(" ")
            row = date2idx[train_date]
            col = date2idx[test_date]
            method = method.rstrip("\n")
            accuracy = np.round(float(accuracy) * 100, 0).astype(int)
            if row == col:
                # same-date validation, shared by all
                for key in result_tables.keys():
                    result_tables[key][row][col].append(accuracy)
            else:
                # cross-date testing with different mapping
                result_tables[method][row][col].append(accuracy)
    
#%%
for key, table in result_tables.items():
    output_file = key + "_final_result.csv"
    temp = np.zeros((9,9)).astype(str)
    for row in range(9):
        for col in range(9):
            mean = np.round(np.mean(table[row][col]), 0).astype(int)
            std = np.round(np.std(table[row][col]), 0).astype(int)
            temp[row][col] = str(mean) + "\u00B1" + str(std)
    columns = [["Test " + date2season[date] for date in dates]]
    index = [["Train " + date2season[date] for date in dates]]
    df = pd.DataFrame(data = temp, columns = columns, index = index)
    df.to_csv(output_file)