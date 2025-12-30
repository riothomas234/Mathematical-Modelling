import numpy as np
import matplotlib.pyplot as plt
import random
import scipy


#Brief:  The following program aims to create a noisy signal composed of an unknown no. of waves of unknown frequency, amplitude, phase
#shift and then to deduce these values by performing a discrete fourier transform

#-----------------------creating signal with unknown number of components of unknown frequency, amplitude, phase shift-----------------------
N = random.randint(1,10)  # number of component waves

freq = np.array([random.randint(1, 400) for i in range(N)])     #choose 5 wave frequencies between 1 and 400 Hz
ang_freq = 2*np.pi*freq   #convert to angular frequency

#construct N complex amplitudes of N waves

real_part = np.random.randint(1, 11, N)  # integers 1-10
imag_part = np.random.randint(1, 11, N)
amplitudes = real_part + 1j * imag_part


# Generate random offset to wave between 1 and 5.
offset = random.randint(1,5)


#signal is sampled at 1000 times per second.
sample_rate =1000

#create linear time range between 0 and 1 seconds of a 1000 intervals.
times = np.linspace(0,1,sample_rate)



#generate random noise and add to signal
noise = np.array([np.random.normal(0,0.5) for i in range(sample_rate)])


#create function initially at 0 for all time intervals.
#set f(x) = sum of C_n * exp(iwt)
i=0
func = np.linspace(0+0j,0+0j, sample_rate)
for w in ang_freq:
    func += amplitudes[i] * np.exp(1j*w*times)
    i += 1

#add noise and offset to wave
func += noise + offset



#------------------------------performing fourier transform on signal-----------------------------


# 'transform' contains information on the amplitudes of all component frequencies
transform = np.fft.fft(func)

#freqs contains the values of the frequencies themselves
freqs = np.fft.fftfreq(len(func), d=1/sample_rate)   #d is frequency spacing - inverse of sampling rate


#only want positive frequencies - the first half of freqs is positive frequencies in increasing order, second half is negative frequencies becoming more negative.
#we extract the first half

half = len(freqs)//2
transform_half = transform[:half]
freqs_half = freqs[:half]


#scipy module to find the indices of frequencies that are greater in amplitude than 1/10th of that of the max amplitude frequency.
# 'peaks, _ = ' creates an array for indices. Note transform_half contains complex amplitudes so we deal with its absolute value.
peaks, _ = scipy.signal.find_peaks(np.abs(transform_half), height = 0.1*np.max(np.abs(transform_half)))

peak_freqs = (freqs_half)[peaks]

#the FFT gives us amplitudes for different frequency wave components scaled up by the number of samples we've taken. Take account for this here.
#note the factor of 2: the fourier transform splits amplitude over positive and negative frequencies. As we've made up our signal from positive frequencies only,
#we need to multiply our amplitudes by 2 to get actual amplitude.
peak_amplitudes = np.abs(2*(len(func))**(-1)*transform_half[peaks])
peak_phases = np.angle(transform_half[peaks])

#Note presence of peak at f=0. We imagine a wave of w=0 as having a constant amplitude. Thus it is our offset... We dont need to multiply by 2 because
#peak at f=0 has no negative reflection

offset = (len(func))**(-1)*np.abs(transform[0])

print('Offset:', offset)
print('Frequencies of component waves:', peak_freqs, '(Hz)')
print('amplitudes of component waves:', peak_amplitudes)
print('phases of component waves:', peak_phases)



#-----------------------Plots---------------------
#Signal plot against time
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")
plt.plot(times, np.abs(func))
plt.show()

#Power spectrum
plt.xlabel("Frequency(Hz)")
plt.ylabel("Power")
plt.plot(freqs_half, (2* (len(func))**(-1)*np.abs(transform_half))**2)    #first half of freqs is the positive values
plt.show()