import tensorflow
print(tensorflow.__version__)
import keras
print(keras.__version__)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


headers = ['Timestamp', 'Signal']
df = pd.read_csv('M:\Master Thesis\TestData\matlabWorkspace\Ring Resonator\\i1.txt',skiprows=range(0,3),names= headers, sep= '\t', float_precision=None)

maxValueIndexObj = df['Signal'].idxmax()

print("Max values of column is at row index position :")
print(maxValueIndexObj)

#The pulse size is twice the max value index
pulseSize = maxValueIndexObj * 2
print(pulseSize)
