#!/usr/bin/env python
"""
The Matched Filter implemenation in frequency domain 
"""
__author__ = "Aleksei Rostov"
__contact__ = "aleksei.rostov@protonmail.com"
__date__ = "2023/03/23"

import numpy as np
import matplotlib.pyplot as plt


N       = 2048      # number of samples
nF1     = 200         # start freq bin
nF2     = 200       # stop  freq bin
Nb      = nF2 - nF1 # spect width
Nt      = 200       # pulse width
Sn      = Nb/Nt/2*N # chirp rate

tn_ax   = np.linspace(0,1,N); # axis in time domain


def main():

    # Input signal pulse
    x       =  np.exp(2*1j*np.pi*(nF1+Sn*tn_ax)*tn_ax)*(np.linspace(0,N-1,N)< Nt)
    # Complex noise with unity power
    n       = (np.random.randn(N) + 1j*np.random.randn(N))/np.sqrt(2)
    # Impulse response
    h       =  np.conjugate(x)
    h       =  np.roll(h[::-1], Nt)
    x_pow   = 20*np.log10(np.sum(np.abs(x))/Nt)
    print("<< Input signal average power is {:3.2f} dB".format(x_pow))
    # noise variance or power
    n_pow   = 10*np.log10(np.var(n))
    print("<< Input noise average power is {:3.2f} dB".format(n_pow))
    
    # FFT input signal and impulse response
    xF      = np.fft.fft(x)
    nF      = np.fft.fft(n)
    hF      = np.fft.fft(h)
    
    # multiplication in frequency domain
    multF_xh= xF*hF
    multF_nh= nF*hF
    
    # IFFT
    y_x     = np.fft.ifft(multF_xh)
    y_n     = np.fft.ifft(multF_nh)
    y       = y_x+y_n
    
    mf_figure(x, h, xF, hF, multF_xh, y_x, "desirable signal")
    mf_figure(n, h, nF, hF, multF_nh, y_n, "white noise")
    
    plt.figure(figsize=(15,10))
    plt.plot((np.abs(y)), '.-')
    plt.title("Matched Filter Output")
    plt.xlabel('Time bins')
    plt.ylabel('Modulus')
    plt.grid()   
    
    plt.show()
   
    
def mf_figure(xT, hT, xF, hF, xFhF, yT, label="signal"):
    plt.figure(figsize=(15,10))
    plt.suptitle("Input is " + label, fontsize=16)
    plt.subplot(2, 2, 1)
    plt.plot(np.abs(xT), '.-g', label="abs(x_in)")
    plt.plot(np.imag(xT), '.-b', label="imag(x_in)")
    plt.plot(np.imag(hT), 'x-r', label="imag(h_mf)")
    plt.xlabel("Time bins")
    plt.ylabel("real()")
    plt.legend(loc='upper right')
    plt.title("MF input")
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(np.abs(xF), '.-b', label="abs(FFT(x_in))")
    plt.plot(np.abs(hF), 'x-r', label="abs(FFT(h_mf))")
    plt.legend(loc='upper right')
    plt.xlabel("Frequency bins")
    plt.ylabel("Modulus")
    plt.title("Signal's spectrum")
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.plot(np.real(xFhF), '.-b', label="real(FFT(x_in)*FFT(h_mf))")
    plt.legend(loc='upper right')
    plt.xlabel("Frequency bins")
    plt.ylabel("real()")
    plt.title("Product of the spectrums")
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(np.abs(yT), '.-b', label="abs(IFFT(FFT(x_in)*FFT(h_mf)))")
    plt.legend(loc='upper right')
    plt.ylabel("abs()")
    plt.xlabel("Time bins")
    plt.title("MF output")
    plt.tight_layout()
    plt.grid()

   

if __name__ == "__main__":
    main()