# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 17:15:10 2020

@author: Eve FLeisig

Extracts IDs of the form CWxxxxxxxxxx or CBxxxxxxxxxx and outputs csv files 
with the xxxxxxxxxx IDs.
"""
import os
import csv



for fname in os.listdir('text strings/text strings'):
    print(fname)
    with open('text strings/text strings/' + fname, 'r', encoding='utf8') as file:
        entries = file.read().split('id":"')[1:]
        #print(entries[1], entries[2])
        
        ids = [entry[:17] for entry in entries]
        #print(ids)
        
        requested_ids = [cur_id[7:] for cur_id in ids if cur_id.startswith('GALE|CW') or cur_id.startswith('GALE|CB')]
        print(requested_ids)
        
        
        with open(fname[:-4] + ".csv", 'w') as output:
            wr = csv.writer(output)
            wr.writerow(requested_ids)
        


