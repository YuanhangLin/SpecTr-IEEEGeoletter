#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 21:42:27 2021

@author: yuanhanglin
"""

import pandas as pd
import numpy as np

#%%
NT = pd.read_csv("no-mapping_final_result.csv")
NT = NT.values[:, 1:].astype(str)

L = pd.read_csv("linear-transformation_final_result.csv")
L = L.values[:, 1:].astype(str)

A = pd.read_csv("affine-transformation_final_result.csv")
A = A.values[:, 1:].astype(str)

DT = pd.read_csv("decision-tree_final_result.csv")
DT = DT.values[:, 1:].astype(str)

RF = pd.read_csv("random-forest_final_result.csv")
RF = RF.values[:, 1:].astype(str)

#%%
tables = {}
for r in range(9):
    tables[r] = {}
    for c in range(9):
        tables[r][c] = []

for r in range(9):
    for c in range(9):
        if r == c:
            temp = NT[r][c].split("±")[0]
            temp = "%.2f" % round(float(temp), 2)
            tables[r][c].append(temp)
        else:
            temp = NT[r][c].split("±")[0]
            temp = "%.2f" % round(float(temp), 2)
            tables[r][c].append(temp)
            temp = A[r][c].split("±")[0]
            temp = "%.2f" % round(float(temp), 2)
            tables[r][c].append(temp)
            temp = RF[r][c].split("±")[0]
            temp = "%.2f" % round(float(temp), 2)
            tables[r][c].append(temp)

#%% table 3
latex = []
for r in range(9):
    temp = []
    for c in range(9):
        if r == c:
            temp.append("\multicolumn{3}{c|}{\cellcolor{lightgray}" + tables[r][c][0] + "}")
        else:
            temp.append(" & ".join(tables[r][c]))
    latex.append(temp)

for i, line in enumerate(latex):
    print("line", i)
    print(" & ".join(line))
    input("press to print the next line")
    
#%% table 2
L_results = []
A_results = []
DT_results = []
RF_results = []
for r in range(9):
    for c in range(9):
        temp = L[r][c].split("±")[0]
        L_results.append(float(temp))

        temp = A[r][c].split("±")[0]
        A_results.append(float(temp))
        
        temp = DT[r][c].split("±")[0]
        DT_results.append(float(temp))
        
        temp = RF[r][c].split("±")[0]
        RF_results.append(float(temp))

print("L: %.2f ± %.2f" %(np.mean(L_results), np.std(L_results)))
print("A: %.2f ± %.2f" %(np.mean(A_results), np.std(A_results)))
print("DT: %.2f ± %.2f" %(np.mean(DT_results), np.std(DT_results)))
print("RF: %.2f ± %.2f" %(np.mean(RF_results), np.std(RF_results)))