#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:54:05 2023

@author: moritz
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 15:57:58 2023

@author: moritz
"""

import cv2
import matplotlib.pyplot as plt

r_means = []
g_means = []
b_means = []

input_video = cv2.VideoCapture('/home/moritz/Downloads/test.mp4')

def plot_data():
    plt.plot(r_means, color='red', label='R')
    plt.plot(g_means, color='green', label='G')
    plt.plot(b_means, color='blue', label='B')
    
    plt.ylabel('Mean brightness')
    plt.xlabel('Frame')
    plt.legend()

    plt.xlim(0, 390)
    plt.ylim(55, 170)
    plt.draw()
    plt.pause(0.00009)

plt.figure().set_figwidth(12)
while 1:
    _, image = input_video.read()

    if image is None:
        plot_data()
        break
    
    r_mean = image[:, :, 0].mean()
    g_mean = image[:, :, 1].mean()
    b_mean = image[:, :, 2].mean()
    
    r_means.append(r_mean)
    g_means.append(g_mean)
    b_means.append(b_mean)
    
    plot_data()
    plt.clf()
    
