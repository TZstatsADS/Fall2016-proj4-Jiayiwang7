import os
import collections
import hdf5_getters
import tables
import numpy as np
from sklearn import mixture
import pandas as pd
from scipy.stats import rankdata


def get_file_path(startdir):
    path = []
    file_name = []
    for dirs, subdirs, files in os.walk(startdir):
        for name in files:
            if name != ".DS_Store":
                file_name.append(name)
                path_s = os.path.join(dirs, name)
                path.append(path_s)
    return(path)


def get_file_name(startdir):
    file_name = []
    for dirs, subdirs, files in os.walk(startdir):
        for name in files:
            if name != ".DS_Store":
                file_name.append(name)
    return(file_name)

def get_list_attr(path_list, attr):
    attr_list = []
    i = 1
    for file in path_list:
        try:
            file_read = hdf5_getters.open_h5_file_read(file)
            #print(hdf5_getters.__getattribute__(attr)(file_read))
            attr_list.append(hdf5_getters.__getattribute__(attr)(file_read))
            
            file_read.close()
            #print 'Finished ' + str(i) + '/2350'
            i += 1
        except:
            print '---- Failed to get ' + file + ' ---- No:' + str(i)
            
            attr_list.append(0)
            i += 1
    return(attr_list)

def stat(attr):
    ls = []
    for l in attr:
        try:
            stats = [np.max(l), np.min(l), np.mean(l), np.var(l), np.ptp(l), np.percentile(l, 25), np.percentile(l, 50), np.percentile(l, 75)]
            ls.append(stats)
        except ValueError:
            ls.append([0] * 8)
    return(ls)


def cmbfat(attr_ls):
    ls0 = attr_ls[0]
    for l in attr_ls[1:]:
        for i in range(len(ls0)):
            ls0[i] += l[i]
    return(ls0)

