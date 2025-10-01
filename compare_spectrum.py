import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import scipy
from planar_wave_extraction import main
from WH_base import wiener_hopf
from scipy.optimize import curve_fit

c = 340.29
# probe_index=97
dt = 1e-6
tmax = 0.067
right=True if len(sys.argv) > 2 and sys.argv[3]=="True" else False
probe_index = 97 if right else 3

# right=False

# tmax = 1.0

fmin = 500  
fmax = 4000
directory = sys.argv[1] if len(sys.argv) >1 else None

if len(sys.argv) >2:
    forcing_file = sys.argv[2]
else:
    forcing_file = None 

ua, pa, fa, ga, f, g, p, u, time, x, y, z = main(plots=False, directory=directory)
# factor = c**2 / 1e4
factor = 1.0
if right:
    uin =  fa[probe_index, :] * factor
    uout = ga[probe_index, :] * factor
else:
    uout = fa[probe_index, :] * factor
    uin = ga[probe_index, :] * factor

timein = time
timeout = time

uin = uin[time < tmax]
uout = uout[time < tmax]
if right:
    dtin = probe_index/100 / c
    dtout = 1/c  + (1 - probe_index/100) / c
else:
    dtin = (1 - probe_index/100) / c
    dtout = 1/c + probe_index / 100 / c


timein = timein[time < tmax] - dtin
timeout = timeout[time < tmax] - dtout


deltat  = time[1]-time[0]

# transfer_function, fWH, h = wiener_hopf(uin, uout, time, L=int(0.3 / c // deltat), deltat=deltat, plot=False, eps_conditioning=1e-10)

N = uin.shape[0]
f = np.fft.fftfreq(N, d=time[1]-time[0])[0:N//2]
Uin_fft = np.fft.fft(uin)[0:N//2]
Uout_fft = np.fft.fft(uout)[0:N//2]

transfer_function = Uout_fft / Uin_fft
fWH = f
# plot signal in and signal out vs time

fig, ax = plt.subplots(figsize=(4,3))

ax.plot(timein, uin, label='Signal In', color='r')
ax.plot(timeout, uout, label='Signal Out', color='b')
if forcing_file is not None:
    forcingdata = np.loadtxt(forcing_file, delimiter =',')
    # forcing = forcing - forcing[0]

    forcingtime = forcingdata[0, :]
    forcingval = forcingdata[1, :] - forcingdata[1, :].mean()
    # indices = np.array(time[time < tmax]//dt, dtype=int)
    # forcing = forcing[indices]
    forcing = np.interp(timein, forcingtime, forcingval)
    ax.plot(timein, forcing, label='Forcing', color='k', linestyle='dashed')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.legend()
ax.grid('both')

plt.tight_layout()
plt.show()
plt.close(fig)

# compute fft of signal in and signal out
# plot fft of both



fig, ax = plt.subplots(figsize=(4,3))

ax.plot(f, np.abs(Uin_fft), label='Signal In', color='r')
ax.plot(f, np.abs(Uout_fft), label='Signal Out', color='b', alpha=0.5)

if forcing_file is not None:
    forcing_fft = np.fft.fft(forcing)[0:N//2]
    ax.plot(f, np.abs(forcing_fft), label='Forcing', color='k', linestyle='dashed')

ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Magnitude')
ax.legend()
ax.grid('both')
ax.set_xlim(0, 1.5 * fmax)


plt.tight_layout()
plt.show()
plt.close(fig)


fig, ax = plt.subplots(figsize=(4,3))

ax.plot(f, np.angle(Uin_fft), label='Signal In', color='r')
ax.plot(f, np.angle(Uout_fft), label='Signal Out', color='b', alpha=0.5)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Phase (radians)')
ax.legend()
ax.grid('both')
ax.set_xlim(0, 1.5 * fmax)

plt.tight_layout()
plt.show()
plt.close(fig)



# compute the transfer function
# plot the TF - gain and phase

TF = Uout_fft / Uin_fft
# TF = Uin_fft / forcing_fft

# TODO: keep in mind what's being plotted
# transfer_function = Uin_fft / forcing_fft

fig, ax = plt.subplots(figsize=(4,3))
if forcing is None:
    ax.plot(f, np.abs(TF), label='Gain (FFT)', color='g')
    ax.plot(fWH, np.abs(transfer_function), label='Gain (W-H)', color='b')

else:
    # TF_F_IN, f_F_IN, h_F_IN = wiener_hopf(forcing, uin, time, L=int(1.0 / c // deltat), deltat=deltat, plot=False, eps_conditioning=1e-10)
    # TF_F_IN, f_F_IN, h_F_IN = wiener_hopf(forcing, uin, time, L=int(0.3 / c // deltat), deltat=deltat, plot=False, eps_conditioning=1e-10)
    # TF_F_OUT, f_F_OUT, h_F_OUT = wiener_hopf(forcing, uout, time, L=int(1.2 / c // deltat), deltat=deltat, plot=False, eps_conditioning=1e-10)

    TF_F_IN = Uin_fft / forcing_fft
    TF_F_OUT = Uout_fft / forcing_fft

    f_F_IN = f
    f_F_OUT = f

    ax.plot(f_F_IN, np.abs(TF_F_IN), label='Forcing -> Input', color='r')
    ax.plot(fWH, np.abs(transfer_function), label='Input -> Output', color='g')
    ax.plot(f_F_OUT, np.abs(TF_F_OUT), label='Forcing -> Output', color='b')

    # ax.plot(fWH, np.abs(transfer_function), label='Forcing->Input', color='g')

def fit_tauIO(fref, vref, tau0=1e-3):
    
    def model(f, tau):
        return np.abs(-1 / (1 + 1j * 2 * np.pi * f * tau))
    
    popt, pcov = curve_fit(model, fref, vref, p0=[tau0], bounds=(0, np.inf))
    tau_best = popt[0]
    vfit = model(fref, tau_best)
    
    return tau_best, vfit, popt, pcov

def fit_tauFI(fref, vref, tau0=1e-3):
    
    def model(f, tau):
        return np.abs(1.0 - 1 / (1 + 1j * 2 * np.pi * f * tau))
    
    popt, pcov = curve_fit(model, fref, vref, p0=[tau0], bounds=(0, np.inf))
    tau_best = popt[0]
    vfit = model(fref, tau_best)
    
    return tau_best, vfit, popt, pcov

def fit_tauInletv2(fref, vref, tau0=1e-3):
    
    def model(f, tau):
        return np.abs(1.0 - 1 / (1 + 1j * 2 * np.pi * f * tau))
    
    popt, pcov = curve_fit(model, fref, vref, p0=[tau0], bounds=(0, np.inf))
    tau_best = popt[0]
    vfit = model(fref, tau_best)
    
    return tau_best, vfit, popt, pcov


args = np.where(np.logical_and(fWH<fmax, fWH > fmin))

tau_best, vfit, popt, pcov = fit_tauIO(fWH[args], np.abs(transfer_function)[args])
print(f'sigma best fit IO: {2 / c / tau_best:.8f}')
print(f'K best fit IO: {2 / tau_best:.8f}')
print(f'f_3dB fit IO: {2 / tau_best / 4 / np.pi :.4f}')
ax.plot(fWH[args], vfit, color='k', linestyle='dashed', linewidth=2, label=rf'$\frac{{-1}}{{1+i \omega \tau}}$' + f', $\\sigma = {2 / c / tau_best:.8f}$')


args = np.where(np.logical_and(f_F_IN<fmax, f_F_IN > fmin))

tau_best, vfit, popt, pcov = fit_tauFI(f_F_IN[args], np.abs(TF_F_IN)[args])
print(f'sigma best fit FI: {2 / c / tau_best:.8f}')
print(f'K best fit FI: {2 / tau_best:.8f}')
print(f'f_3dB fit FI: {2 / tau_best / 4 / np.pi :.4f}')
ax.plot(fWH[args], vfit, color='k', linestyle='dashdot', linewidth=2, label=rf'$1 + \frac{{-1}}{{1+i \omega \tau}}$' + f', $\\sigma = {2 / c / tau_best:.8f}$')


# # Compute cutoff frequency
# f_c = 1 / (2 * np.pi * tau_best)

# # Sharp cutoff (brick-wall)

# fWH = np.linspace(f_c/10, fmax, 5000)
# sharp_cutoff = np.ones(fWH.shape)
# sharp_cutoff[np.where(fWH > f_c)] = f_c / fWH[np.where(fWH > f_c)]
# ax.plot(
#     fWH,
#     sharp_cutoff,
#     color='k',
#     linestyle=':',
#     linewidth=2,
#     label=rf'Sharp cutoff'
# )


# print(slope_asymptote)
# ax.plot(f, np.angle(TF), label='Phase', color='m')
ax.set_xlabel('Frequency (Hz)')
# ax.set_ylabel('Magnitude Uout/Uin')
ax.set_ylabel('Magnitude')

ax.legend()
ax.grid(visible=True, which='major', color='k', linestyle='-')
ax.grid(visible=True, which='minor', color='k', linestyle='--', alpha=0.5)

ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim(1e2, fmax)

plt.tight_layout()
plt.show()
plt.close(fig)
