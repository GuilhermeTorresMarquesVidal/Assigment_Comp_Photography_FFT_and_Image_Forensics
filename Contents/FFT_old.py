import numpy as np

def fft(img):
    
    M, N = img.shape
    
    C = np.zeros((M, N))
    S = np.zeros((M, N))
    
    ns = np.arange(M)
    wave = 2.0 * np.pi * ns / M
    
    for k in range(M):
        x = k * wave
        C[k,:] = np.cos(x)
        S[k,:] = np.sin(x)
        
    img_out = 1/M * C.dot(img) - 1j/M * S.dot(img)
    
    return img_out

def ifft(img):
    
    M, N = img.shape
    
    C = np.zeros((M, N))
    S = np.zeros((M, N))
    
    ns = np.arange(M)
    wave = 2.0 * np.pi * ns / M
    
    for k in range(M):
        x = k * wave
        C[k,:] = np.cos(x)
        S[k,:] = np.sin(x)
    img_out = 1/M * C.dot(img) + 1j/M * S.dot(img)
    
    return img_out

def fft2d(img):
    
    return np.transpose(fft(np.transpose(fft(img))))

def ifft2d(img):
    
    return np.transpose(ifft(np.transpose(ifft(img))))