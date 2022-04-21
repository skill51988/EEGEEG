#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 16:21:27 2022

@author: hayun
"""


fig, ax = plt.subplots(ncols=4, nrows = 2, figsize=(8, 4), gridspec_kw=dict(top=0.9),
                       sharex=True, sharey=True)

reject = dict(eeg = 150e-6)

# for i in range(8):
    # epochs[i].drop_bad(reject = reject)

for i in range(4):
    epochs[i].average().plot(axes = ax[0][i])
    # fig[i].set_title('{}'.format(i))

for i in range(4):
    epochs[i+4].average().plot(axes = ax[1][i])
    # fig[i+4].set_title('{}'.format(i+4))