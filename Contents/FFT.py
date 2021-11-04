import numpy as np

def FFT(signal):

    N = len(signal)
    if(N == 1):
        return signal + 0j
    
    n = np.arange(N)
    w = np.exp(2j*np.pi*n/N)
    
    signal_even, signal_odd = signal[::2], signal[1::2]
    y_even, y_odd = FFT(signal_even), FFT(signal_odd)

    y = np.zeros_like(signal, dtype = np.cdouble)
    
    for i in range(int(N/2)):
        t = w[i]*y_odd[i]
        y[i] = y_even[i] + t
        y[i+int(N/2)] = y_even[i] - t

    return y

def IFFT(signal):
    
    N = len(signal)
    if(N == 1):
        return signal + 0j
    
    n = np.arange(N)
    w = np.exp(2j*np.pi*n/N)
    
    signal_even, signal_odd = signal[::2], signal[1::2]
    y_even, y_odd = FFT(signal_even), FFT(signal_odd)

    y = np.zeros_like(signal, dtype = np.cdouble)
    
    for i in range(int(N/2)):
        t = w[i]*y_odd[i]
        y[i] = y_even[i] + t
        y[i+int(N/2)] = y_even[i] - t
    
    return y/N

def FFT2D(img):

    M, N = img.shape
    
    img_temp = np.zeros_like(img, dtype = np.cdouble)
    img_frequency = np.zeros_like(img, dtype = np.cdouble)

    for i in range(M):
        img_temp[i,:] = FFT(img[i,:])

    for i in range(N):
        img_frequency[:,i] = FFT(img_temp[:,i])

    return img_frequency

def IFFT2D(img_frequency):

    M, N = img_frequency.shape
    
    img_temp = np.zeros_like(img_frequency, dtype = np.cdouble)
    img = np.zeros_like(img_frequency, dtype = np.cdouble)

    for i in range(M):
        img_temp[i,:] = IFFT(img_frequency[i,:])

    for i in range(N):
        img[:,i] = IFFT(img_temp[:,i])

    return img