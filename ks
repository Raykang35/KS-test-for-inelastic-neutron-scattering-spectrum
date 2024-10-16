########## INS data Kolmogorov-Smirnov test ############ 
## References: 
# [1] Numerical Recipes in C by William H. Press, Brian P. Flannery, Saul A. Teukolsky, and William T. Vetterling 
# [2] William H. Press, Saul A. Teukolsky; Kolmogorov‐Smirnov Test for Two‐Dimensional Data: How to tell whether a set of (x,y) data 
# paints are consistent with a particular probability distribution, or with another data set. Comput. Phys. 1 July 1988; 2 (4): 74–77.

# Author: Ray
# Created date: 2024/09/28
# Modified date: 2024/10/12

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

def getXY(filename, type="sim"):
    """ Extract frequency and intensity as xx and yy and return as array
    Args:
    filename: The name of the data file.
    type: Simulation file (sim) or Experimental file (exp)
    ###################################################################
    Return:
    xx: frequency [cm-1]
    yy: intensity [a.u.]
    """
    xx = []
    yy = []
    largest = 0
    with open(filename) as f: # Read the data file
        lines = f.readlines()
    
    if type == 'sim': # Simulation file (DFT, DFTB, MD)
        for line in lines:
            if line[1] != "#": # Skip first 3 rows 
                newData = line.split(',')
                newData = [float(val) for val in newData]
                xx.append(newData[0])
                yy.append(newData[1]+newData[2])   
    
    elif type == 'exp': # Experimental file
        flag = 0 # flag is the number to set the Q, 0 is high Q (backscattering detector) data; 1 is low Q (forward scattering detector) data, 2 is the average of high and low Q data
        isMev = True # Here we assume the frequency unit is meV (it can be wavenumber)
        for line in lines:
            if line != '\n':
                data = " ".join(line.split(',')).split()
                if len(data) == 3 and flag == 1: # Picking low Q data
                    
                    if float(data[0]) < -10:
                        isMev = False # X unit is wavenumber here
                    
                    if float(data[0]) > 1:
                        xx.append(float(data[0]))
                        yy.append(float(data[1]))	
                        if x[-1] > 100 and y[-1] > largest:
                            largest = y[-1]
                            
                elif len(data) == 1:
                    flag = flag + 1
        if isMev:
            xx = [val * 8.06554 for val in x] # Convert meV to cm-1
    
    return xx, yy

def quadct(x, y, xx, yy):
    """
    Given an origin (x, y), and an array of nn points with coordinates xx[1..nn] and yy[1..nn], count how many of them are in each quadrant around the origin, and return the normalized        fractions. Quadrants are labeled alphabetically, counterclockwise from the upper right. Used by ks2d1s and ks2d2s.
    Args:
    x: Reference x point 
    y: Reference y point
    xx: The x array for the spectrum
    yy: The y array for the spectrum
    ###########################
    Return:
    f1 (float): Fraction of total nn points in I quadrant
    f2 (float): Fraction of total nn points in II quadrant
    f3 (float): Fraction of total nn points in III quadrant
    f4 (float): Fraction of total nn points in IV quadrant
    """
    n1 = 0 # The number of points in quadrant I
    n2 = 0 # The number of points in quadrant II
    n3 = 0 # The number of points in quadrant III
    n4 = 0 # The number of points in quadrant IV
    nn = len(xx) # The number of points in xx and yy array
    # The loop below is to categorize the quadrants for all nn points.
    for k in range(1,nn):
        if yy[k] > y and xx[k] > x:
            n1 += 1
        elif yy[k] > y and xx[k] < x:
            n2 += 1
        elif yy[k] < y and xx[k] < x:
            n3 += 1
        elif yy[k] < y and xx[k] > x:
            n4 += 1
            
    ff = 1.0 / nn
    f1 = ff * n1
    f2 = ff * n2
    f3 = ff * n3
    f4 = ff * n4

    return f1, f2, f3, f4
    
def probks(alam):
    """
    Kolmogorov-Smirnov probability function Q_KS. Equation 14.3.7 of Numerical Recipe in C
    Args:
    alam (float): The K-S statistic (with some coefficient) value
    #####################
    Return:
    q (float): The significance level (The number is in between 0 and 1)
    """
    
    coeff = 2.0
    a2 = -2.0 * alam * alam
    q = 0
    for j in range(1,100):
        if j % 2 == 0:
            term = (-1) * coeff * np.exp(a2 * j * j) # Alternating signs in term
            
        else:
            term = coeff * np.exp(a2 * j * j)
        
        q += term

        if np.abs(term) <= 1e-10: # Need to check convergence, the number should be small enough 
            return q
    
    return 1.0 # When alam = 0, q should equal to 1!
    
def ks2d2s(filelist, typelist):
    """ Run kolmogorov-smirnov test to get the similarity between 2 spectrums, 2d means 2 dimensional and 2s means 2 samples.
    Two-dimensional Kolmogorov-Smirnov test on two samples. Given the x and y coordinates of the first sample as n1 values in arrays x1[1..n1] and y1[1..n1], 
    and for the second sample, n2 values in arrays x2 and y2, this routine returns the two-dimensional, two-sample K-S statistic as d, and its significance level as prob. 
    Small values of prob show that the two samples are significantly different. 
    (Note that the test is slightly distribution-dependent, so prob is only an estimate.)
    Args:
    filelist (list): A list saves the name of the files (Now can only have 2 files in the list)
    typelist (list): A list saves the type of the files
    ##############
    Return:
    d: K-S statistic 
    prob: Significance level (Small value shows the two samples are significantly different)
    """
    
    xx1, yy1 = getXY(filelist[0])
    xx2, yy2 = getXY(filelist[1])

    # Use points in first spectrum as origins
    d1 = 0.0
    n1 = len(xx1)
    for j in range(1, n1):
        f1, f2, f3, f4 = quadct(xx1[j],yy1[j],xx1,yy1)
        g1, g2, g3, g4 = quadct(xx2[j],yy2[j],xx2,yy2)
        d1 = np.fmax(d1,abs(f1-g1))
        d1 = np.fmax(d1,abs(f2-g2))
        d1 = np.fmax(d1,abs(f3-g3))
        d1 = np.fmax(d1,abs(f4-g4))

    # Then, use points in second spectrum as origins
    d2 = 0.0
    n2 = len(xx2)
    for j in range(1, n2):
        f1, f2, f3, f4 = quadct(xx2[j],yy2[j],xx1,yy1)
        g1, g2, g3, g4 = quadct(xx2[j],yy2[j],xx2,yy2)
        d2 = np.fmax(d2,abs(f1-g1))
        d2 = np.fmax(d2,abs(f2-g2))
        d2 = np.fmax(d2,abs(f3-g3))
        d2 = np.fmax(d2,abs(f4-g4))
    
    # Average K-S statistics
    d = 0.5 * (d1 + d2)
    N = (n1 * n2) / (n1 + n2) # effective number of data points
    sroot_N = np.sqrt(N)
    r1, p1 = pearsonr(xx1,yy1) # Get the linear correlation coefficient r1
    r2, p2 = pearsonr(xx2,yy2) # Get the linear correlation coefficient r2
    rr = np.sqrt(1 - 0.5 * (r1 * r1 + r2 * r2))
    alam = sroot_N * d / (1 + (0.25 - 0.75 * sroot_N))
    prob = probks(alam)

    return round(d,3), round(prob,3)
