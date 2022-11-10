from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import hilbert


df01 = pd.DataFrame(columns = ['Timestamp', 'Signal'])

for t in range(8000):
    y = np.sin(18*0.01*t) * np.exp(-0.1 * 0.01 * t)
    z = np.sin(15*0.01*t) * np.exp(-0.1*0.01*t)
    x = np.sin(12*0.01*t) * np.exp(-0.1*0.01*t)
    p = y + z + x
    df01 = pd.concat([df01, pd.DataFrame.from_records([{ 'Timestamp':t, 'Signal':p }])], ignore_index=True)

time_col = df01['Timestamp'].to_numpy()
signal_col = df01['Signal'].to_numpy()

def exp(t, a, b):
    return a * np.exp(-b*(t))

def find_peaks(x, y):
    peak_x = []
    peak_vals = []
    for i in range(len(y)):
        if i == 0:
            continue
        if i == len(y) - 1:
            continue
        if (y[i-1] < y[i]) and (y[i+1] < y[i]):
            peak_x.append(x[i])
            peak_vals.append(y[i])
    return np.array(peak_x), np.array(peak_vals)

peak_x, peak_y = find_peaks(time_col, signal_col)


#envelop calculation
analytic_signal = hilbert(signal_col)
amplitude_envelope = np.abs(analytic_signal)
fig, (ax0, ax1) = plt.subplots(nrows=2)
ax0.plot(time_col, signal_col, label='signal')
ax0.plot(time_col, amplitude_envelope, label='envelope')

#decay calculation
popt, pcov = curve_fit(exp, peak_x, peak_y)
print(popt, pcov)
plt.plot(time_col,signal_col)
plt.plot(peak_x, exp(peak_x, *popt))
plt.show()
