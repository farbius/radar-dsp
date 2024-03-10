import numpy as np
import argparse
import os
import sys
import matplotlib.pyplot as plt
from scipy.signal import windows
from scipy.signal import find_peaks
from scipy.stats  import linregress
from scipy.signal import spectrogram

__author__ = "Aleksei Rostov"
__contact__ = "aleksei.rostov@protonmail.com"
__date__ = "2023/11/23"

F0_Hz, F1_Hz    =  2.463e6, 2.463e6
Fclk_Hz = 100.0e6
Namp    = 2**16
Nlut    = 2**10
Ndds    = 2**12
Nfft    = 128

dt      = 1/Fclk_Hz
F_axis  = np.linspace(-.5, .5, Ndds)*Fclk_Hz/1e6
x       = np.linspace(0, Ndds-1, Ndds)
    
updating_limits = False

# Function to handle the zoom event
def on_xlim_change(event_ax, axs):
    global updating_limits
    if updating_limits:
        return
    updating_limits = True
    if event_ax == axs[0]:
        axs[1].set_xlim(axs[0].get_xlim())
    elif event_ax == axs[1]:
        axs[0].set_xlim(axs[1].get_xlim())
    updating_limits = False


def main():
    
    print(f'<< Start Application')
    trunc_t = dds_param(F0_Hz, F1_Hz, Fclk_Hz, 'truncated', Namp, Nlut, Ndds)
    cmpx_lin, phase_lut = trunc_t.dds_lut()
    
    dithr_t = dds_param(F0_Hz, F1_Hz, Fclk_Hz, 'dithered' , Namp, Nlut, Ndds)
    cmpx_dit, phase_dit = dithr_t.dds_lut()
    
    taylr_t = dds_param(F0_Hz, F1_Hz, Fclk_Hz, 'taylor'   , Namp, Nlut, Ndds)
    cmpx_tlr, _ = taylr_t.dds_lut()
    
    golds_t = dds_param(F0_Hz, F1_Hz, Fclk_Hz, 'none'     , Namp, Nlut, Ndds)
    cmpx_gld, phase_acc = golds_t.dds_lut()
    
    win         = windows.blackmanharris(Ndds)
    xF_trunc    = np.fft.fftshift(np.fft.fft(win*cmpx_lin))
    spec_trc    = 10*np.log10(np.abs(xF_trunc)**2 / Ndds)
    spec_trc    = spec_trc - np.max(spec_trc)
    
    peaks, _    = find_peaks(spec_trc, height=-200, distance=20)
    peaks_trc   = peaks[np.argsort(-spec_trc[peaks])[:2]]
    
    xF_dith     = np.fft.fftshift(np.fft.fft(win*cmpx_dit))
    spec_dth    = 10*np.log10(np.abs(xF_dith)**2 / Ndds)
    spec_dth    = spec_dth - np.max(spec_dth)
    
    peaks, _    = find_peaks(spec_dth, height=-200, distance=20)
    peaks_dth   = peaks[np.argsort(-spec_dth[peaks])[:2]]
    
    xF_taylor   = np.fft.fftshift(np.fft.fft(win*cmpx_tlr))
    spec_tlr    = 10*np.log10(np.abs(xF_taylor)**2 / Ndds)
    spec_tlr    = spec_tlr - np.max(spec_tlr)
    
    peaks, _    = find_peaks(spec_tlr, height=-200, distance=20)
    peaks_tlr   = peaks[np.argsort(-spec_tlr[peaks])[:2]]
    
    xF_gold     = np.fft.fftshift(np.fft.fft(win*cmpx_gld))
    spec_gold   = 10*np.log10(np.abs(xF_gold)**2 / Ndds)
    spec_gold   = spec_gold - np.max(spec_gold)
    
    plt.figure(figsize=(10, 6))
    
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(x, phase_lut, 'o-b', fillstyle='none', label='truncated')
    ax1.plot(x, phase_dit, '^-r', fillstyle='none', label='dithered')
    ax1.plot(x, phase_acc, '.-y', label='gold')
    ax1.legend(loc='upper right')
    ax1.set_title('Quantizer Input')
    ax1.grid()

    
    ax2 = plt.subplot(2, 2, 3)
    ax2.plot(x, phase_dit-phase_acc, '.-r', fillstyle='none', label='Phase Dithered  Error')
    ax2.plot(x, phase_lut-phase_acc, '.-b', fillstyle='none', label='Phase Truncated Error')
    ax2.legend(loc='upper right')
    ax2.set_title('Phase Error')
    ax2.set_ylim([-2, 2])
    ax2.grid()
    
    initial_xlim = (0, Ndds)
    ax1.set_xlim(initial_xlim)
    ax2.set_xlim(initial_xlim)
    
    axs = (ax1, ax2)

    # Connect the xlim_changed event between the subplots
    cid1 = ax1.callbacks.connect('xlim_changed', lambda event: on_xlim_change(ax1, axs))
    cid2 = ax2.callbacks.connect('xlim_changed', lambda event: on_xlim_change(ax2, axs))
    
    plt.subplot(1, 2, 2)
    plt.plot(F_axis, spec_trc, 'o-b', fillstyle='none', label="dds truncated")
    plt.plot(F_axis[peaks_trc[1]], spec_trc[peaks_trc[1]], 'og', label=f'SFDR = {spec_trc[peaks_trc[1]]:.2f} dB')
    plt.plot(F_axis, spec_dth, '^-r', fillstyle='none', label="dds dither")
    plt.plot(F_axis[peaks_dth[1]], spec_dth[peaks_dth[1]], 'og', label=f'SFDR = {spec_dth[peaks_dth[1]]:.2f} dB')
    plt.plot(F_axis, spec_tlr, 'x-m', fillstyle='none', label="dds taylor")
    plt.plot(F_axis[peaks_tlr[1]], spec_tlr[peaks_tlr[1]], 'og', label=f'SFDR = {spec_tlr[peaks_tlr[1]]:.2f} dB')
    plt.plot(F_axis, spec_gold, '.-y', fillstyle='none', label="gold signal")
    plt.legend(loc='upper right')
    plt.title('SFDR for Direct Digital Synthesizer')
    plt.xlabel('Frequency, MHz')
    plt.ylabel('Powerdensity, dB')
    plt.grid()

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    
    print(f"<< chirp signal analysis")
    # chirp signal    
    chirp0_t    = dds_param(2.0e6, 12.0e6, Fclk_Hz, 'taylor'     , Namp, Nlut, Ndds)
    tlr_lfm, _  = chirp0_t.dds_lut()
    
    freq_tlr    = np.diff(np.unwrap(np.angle(tlr_lfm))) / dt / (2*np.pi)
    s_tlr, inter_tlr, r_val_tlr, p_val_tlr, std_err_tlr = linregress(x[1:]*dt, freq_tlr)
    tlr_lin     = f"Taylor correction\nSlope: {s_tlr:.2f}\nIntercept: {inter_tlr:.2f}\nR-squared: {r_val_tlr**2:.2f}\nStd Err: {std_err_tlr:.2f}"
    
    chirp0_t    = dds_param(2.0e6, 12.0e6, Fclk_Hz, 'dithered'     , Namp, Nlut, Ndds)
    dit_lfm, _  = chirp0_t.dds_lut()
    
    chirp0_t    = dds_param(2.0e6, 12.0e6, Fclk_Hz, 'none'     , Namp, Nlut, Ndds)
    gld_lfm, _  = chirp0_t.dds_lut()
    
    freq_dit    = np.diff(np.unwrap(np.angle(dit_lfm))) / dt / (2*np.pi)
    s_dit, inter_dit, r_val_dit, p_val_dit, std_err_dit = linregress(x[1:]*dt, freq_dit)
    tlr_dit     = f"Phase Dithering\nSlope: {s_dit:.2f}\nIntercept: {inter_dit:.2f}\nR-squared: {r_val_dit**2:.2f}\nStd Err: {std_err_dit:.2f}"
    
    xF_dith     = np.fft.fftshift(np.fft.fft(dit_lfm))
    spec_dth    = 10*np.log10(np.abs(xF_dith)**2 / Ndds)
    spec_dth    = spec_dth - np.max(spec_dth)
    
    xF_taylor   = np.fft.fftshift(np.fft.fft(tlr_lfm))
    spec_tlr    = 10*np.log10(np.abs(xF_taylor)**2 / Ndds)
    spec_tlr    = spec_tlr - np.max(spec_tlr)
    
    xF_gold     = np.fft.fftshift(np.fft.fft(gld_lfm))
    spec_gold   = 10*np.log10(np.abs(xF_gold)**2 / Ndds)
    spec_gold   = spec_gold - np.max(spec_gold)

    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(x[1:]*dt/1e-6, freq_tlr/1e6, 'x-m', fillstyle='none', label='Inst Freq Taylor')
    plt.plot(x[1:]*dt/1e-6, freq_dit/1e6, '^-r', fillstyle='none', label='Inst Freq Dither')
    plt.plot(x[1:]*dt/1e-6, (s_tlr * x[1:]*dt + inter_tlr)/1e6, 'o-y', fillstyle='none', label='Inst Freq fitted (Taylor)')
    
    plt.text(0, 40, tlr_lin, fontsize=10, color='black', verticalalignment='top')
    plt.text(0, 0, tlr_dit, fontsize=10, color='black', verticalalignment='top')
    plt.xlabel('Time, usec')
    plt.ylabel('Instantaneous Frequency, MHz')
    plt.title('Chirp signal analysis: linear regression')
    plt.ylim([-Fclk_Hz/1e6/2, Fclk_Hz/1e6/2])
    plt.legend()
    plt.grid()
    
    plt.subplot(2, 2, 3)
    plt.plot(x*dt/1e-6, np.abs(tlr_lfm)/Namp, '.-m', fillstyle='none', label='Taylor')
    plt.plot(x*dt/1e-6, np.abs(dit_lfm)/Namp, '.-r', fillstyle='none', label='Dither')
    plt.title('Signal Envelope')
    plt.xlabel('Time, usec')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    
    plt.subplot(1, 2, 2)
    plt.plot(F_axis, spec_tlr, 'x-m', fillstyle='none', label="dds taylor")
    plt.plot(F_axis, spec_dth, '^-r', fillstyle='none', label="dds dither")
    plt.plot(F_axis, spec_gold,'.-y', fillstyle='none', label="gold signal")
    plt.title('Normalized chirp spectrum')
    plt.xlabel('Frequency, MHz')
    plt.ylabel('Powerdensity, dB')
    plt.legend()
    plt.grid()
    plt.tight_layout()

    print(f'<< End Application')
    plt.show()



class dds_param:

    A = 1664525
    C = 1013904223
    M = 2**31
    LOW = -0.5
    UPP =  0.5
    # Seed for the LCG
    SEED= 1024
    def __init__(self, F0_Hz, F1_Hz, Fclk_Hz, SFDR, Namp, Nlut, Ndds):
        self.F0_Hz  = F0_Hz
        self.F1_Hz  = F1_Hz
        self.Fclk_Hz= Fclk_Hz
        self.SFDR   = SFDR
        self.Namp   = Namp
        self.Nlut   = Nlut
        self.Ndds   = Ndds
        
    def _acc_phase(self):
        phase = np.zeros(self.Ndds, dtype=float)
    
        for idx in range(self.Ndds):
            t = idx / self.Fclk_Hz
            f_t = self.F0_Hz + (self.F1_Hz - self.F0_Hz) / (2 / self.Fclk_Hz * self.Ndds) * t
            phase[idx] = f_t / self.Fclk_Hz * self.Nlut * idx
        return phase
    
    def _lcg_float(self):
        result = np.empty(self.Ndds, dtype=float)
        for i in range(self.Ndds):
            self.SEED = (self.A * self.SEED + self.C) % self.M
            result[i] = self.SEED / (self.M / (self.UPP - self.LOW)) + self.LOW
        return result
        
    def dds_lut(self):
        phase_acc = self._acc_phase()
        lut_re    = np.floor(self.Namp*np.cos(2*np.pi*np.linspace(0, 1, self.Nlut, endpoint=False)))
        lut_im    = np.floor(self.Namp*np.sin(2*np.pi*np.linspace(0, 1, self.Nlut, endpoint=False)))
        
        if self.SFDR == 'truncated':
            phase_acc   = phase_acc % self.Nlut
            phase_lut   = np.floor(phase_acc).astype(np.uint16)
            cmpx_out    = lut_re[phase_lut] + 1j*lut_im[phase_lut]
            
        if self.SFDR == 'dithered':
            phase_dit   = phase_acc + self._lcg_float()
            phase_dit   = phase_dit % self.Nlut
            phase_lut   = np.floor(phase_dit).astype(np.uint16)
            cmpx_out    = lut_re[phase_lut] + 1j*lut_im[phase_lut]

        if self.SFDR == 'taylor':# check if there is function
            phase_acc   = phase_acc % self.Nlut
            phase_lut   = np.floor(phase_acc).astype(np.uint16)
            phase_x0    = phase_lut
            phase_x1    =(phase_lut + 1) % self.Nlut

            sin_x0      = lut_im[phase_x0]
            cos_x0      = lut_re[phase_x0]

            sin_x1      = lut_im[phase_x1]
            cos_x1      = lut_re[phase_x1]

            d_sin       = (sin_x1 - sin_x0)
            d_cos       = (cos_x1 - cos_x0)

            lsb, _      = np.modf(phase_acc)
            sin_apprx   = d_sin*lsb + sin_x0
            cos_apprx   = d_cos*lsb + cos_x0
            cmpx_out    = cos_apprx+ 1j*sin_apprx
            
        if self.SFDR == 'none':
            phase_lut   = phase_acc % self.Nlut
            cmpx_out    = self.Namp*np.exp(1j*2*np.pi*(phase_lut/self.Nlut))          
        
        return cmpx_out, phase_lut
    
if __name__ == "__main__":
    main()