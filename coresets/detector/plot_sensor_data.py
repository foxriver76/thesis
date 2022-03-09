#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 19:08:16 2020

@author: Moritz Heusinger <moritz.heusinger@gmail.com>
"""

import matplotlib.pyplot as plt
import numpy as np

rssi_values = [-80,-82,-84,-71,-79,-85,-76,-71,-73,-76,-72,-73,-80,-74,-76,-73,-68,-76,-82,-74,-83,-77,-75,-77,-79,-78,-77,-82,-77,-81,-75,-78,-76,-77,-78,-75,-76,-84,-74,-80,-75,-78,-81,-76,-81,-85,-75,-84,-76,-76,-77,-82,-82,-79,-81,-83,-77,-81,-78,-86,-85,-81,-76,-82,-86,-79,-77,-81,-85,-84,-85,-84,-67,-59,-65,-54,-47,-47,-46,-49,-47,-47,-47,-48,-62,-46,-50,-58,-46,-51,-45,-50,-45,-58,-57,-50,-58,-46,-50,-51,-44,-47,-43,-43,-51]
rssi_values_after_drift = rssi_values.copy()

for i in range(len(rssi_values)):
    if rssi_values[i] >= -65:
        rssi_values[i] = np.nan
        
for i in range(len(rssi_values_after_drift)):
    if rssi_values_after_drift[i] <= -70:
        rssi_values_after_drift[i] = np.nan

plt.plot(rssi_values, color='blue', linestyle=':', linewidth='2')
plt.plot(rssi_values_after_drift, color='orange', linestyle='-', linewidth='2')
plt.xlabel('timestep')
plt.ylabel('rssi')
plt.savefig('../figures/rssi_localization.eps')
plt.show()