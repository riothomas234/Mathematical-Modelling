

import sys

print(sys.executable)
import matplotlib.pyplot as plt
import random
import numpy as np

freq = np.array([random.randint(1,400) for i in range(6)])
ang_freq = 2*np.pi*freq
amplitude = 5
phase =np.pi/3
offset = 1

split=1000

times = np.linspace(0,1,split)  #1 second, split is sampling rate
noise = np.array([np.random.normal(0,0.5) for i in range(split)])

func=np.linspace(0,0,split)
for w in ang_freq:
    func += amplitude*np.sin(w*times+phase)
func+= offset + noise


plt.plot(times, func)
plt.show()

#fourier transform - what type? can't integrate my plotted function - isnt continuous. just doing FFT is hard, because i dont know how to do on a list.
#any libraries i can use?

transform = np.fft.fft(func)
freqs = np.fft.fftfreq(len(func), d=1/split)   #d is sample spacing - inverse of sampling rate - why??

half = len(freqs)//2

plt.plot(freqs[:half], np.absolute(transform[:half]))
plt.show()
