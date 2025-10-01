import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import scipy
from planar_wave_extraction import main as PWE
from WH_base import wiener_hopf
from scipy.optimize import curve_fit
from scipy.ndimage import shift

c = 340.29
dt = 1e-6
tmin_reflection = 0.0640
tmax_reflection = 0.0675

fmin = 250
fmax = 5000
fmax_interp = 4000

probe_index=50
forcing_file_right = 'BC/Wavelet_100_4000_0.20000000.U'
forcing_file_left = 'BC/Wavelet_100_4000_0.20000000.U'

dir = sys.argv[1] if len(sys.argv) > 1 else 'TEMPORAL_T03_CORR'
print('using dir:', dir)


ua, pa, fa, ga, f, g, p, u, time, x, y, z = PWE(plots=False, directory=dir)
deltat = time[1]-time[0]

# fa = fa[probe_index, :]
# ga = ga[probe_index, :]

fa = f[probe_index, :]
ga = g[probe_index, :]

deltat_shift_right = x[probe_index] / c
deltat_shift_left = (1-x[probe_index]) / c

    # transfer_function, fWH, h = wiener_hopf(uin, uout, time, L=int(0.3 / c // deltat), deltat=deltat, plot=False, eps_conditioning=1e-10)

N = fa.shape[0]
f = np.fft.fftfreq(N, d=time[1]-time[0])[0:N//2]
fa_fft = np.fft.fft(fa)[0:N//2]
ga_fft = np.fft.fft(ga)[0:N//2]

forcingdata_right = np.loadtxt(forcing_file_right, delimiter =',')
forcingtime_right = forcingdata_right[0, :]
forcingval_right = forcingdata_right[1, :]

forcingdata_left = np.loadtxt(forcing_file_left, delimiter =',')
forcingtime_left = forcingdata_left[0, :]
forcingval_left = forcingdata_left[1, :]



# indices = np.array(time[time < tmax]//dt, dtype=int)
# forcing = forcing[indices]
forcing_right = np.interp(time, forcingtime_right, forcingval_right)
forcing_left = np.interp(time, forcingtime_left, forcingval_left)

forcing_fft_right = np.fft.fft(forcing_right)[0:len(forcing_right)//2]
forcing_fft_left = np.fft.fft(forcing_left)[0:len(forcing_left)//2]

forcing_freq = np.fft.fftfreq(len(forcing_right), d=time[1]-time[0])[0:len(forcing_right)//2]


fig, ax = plt.subplots(figsize=(4,3))

ax.plot(time - deltat_shift_right, fa, label='$f$', color='r')
ax.plot(time - deltat_shift_left, ga, label='$g$', color='b')
ax.plot(time, forcing_right, label='Inlet forcing', color='r', linestyle='dashed')
ax.plot(time, forcing_left, label='Outlet forcing', color='b', linestyle='dashed')

    # ax.plot(timeout, uout, label='Signal Out', color='b')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.legend()
ax.grid(visible=True, which='major', color='k', linestyle='-')
ax.grid(visible=True, which='minor', color='k', linestyle='--', alpha=0.5)


plt.tight_layout()
plt.show()
plt.close(fig)

fig, ax = plt.subplots(figsize=(4,3))

ax.plot(f, np.abs(fa_fft), label='$f$', color='r')
ax.plot(f, np.abs(forcing_fft_right), label='Inlet forcing', color='k', linestyle='dashed' , alpha=0.5)

ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Magnitude')
ax.legend()
ax.grid('both')
ax.set_xlim(0, 1.5 * fmax)


plt.tight_layout()
plt.show()
plt.close(fig)

fig, ax = plt.subplots(figsize=(4,3))

ax.plot(f, np.abs(ga_fft), label='$g$', color='b')
ax.plot(f, np.abs(forcing_fft_left), label='Outlet forcing', color='k', linestyle='dashed', alpha=0.5)

ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Magnitude')
ax.legend()
ax.grid('both')
ax.set_xlim(0, 1.5 * fmax)


plt.tight_layout()
plt.show()
plt.close(fig)


injection_coeff_inlet = fa_fft / forcing_fft_right
injection_coeff_outlet = ga_fft / forcing_fft_left
# injection_coeff_inlet, f, _ = wiener_hopf(fa, forcing_left, time, L=int(1.0 / c // deltat), deltat=deltat, plot=False, eps_conditioning=1e-10)
# injection_coeff_outlet, _, _ = wiener_hopf(ga, forcing_right, time, L=int(1.0 / c // deltat), deltat=deltat, plot=False, eps_conditioning=1e-10)



fig, ax = plt.subplots(figsize=(4,3))

def fit_tauIO(fref, vref, K0=1.0, A0=1.0, Amax=1.000001):
    
    def model(f, K, A):
        return np.abs(-1 * A / (1 + 1j * 4 * np.pi * f / K))
    
    # popt, pcov = curve_fit(model, fref, vref, p0=[K0, A0], bounds=([0.0, 0.99], [np.inf, Amax]))
    popt, pcov = curve_fit(model, fref, vref, p0=[K0, A0], bounds=([0.0, 0.99], [np.inf, 1.00000001]))

    args_best = popt
    vfit = model(fref, *args_best)
    
    return args_best, vfit, popt, pcov

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



ax.plot(f, np.abs(injection_coeff_inlet), label=r'$\hat{t}_{inlet}$', color='r')
ax.plot(f, np.abs(injection_coeff_outlet), label=r'$\hat{t}_{outlet}$', color='b')

ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Gain')

ax.legend()
ax.grid(visible=True, which='major', color='k', linestyle='-')
ax.grid(visible=True, which='minor', color='k', linestyle='--', alpha=0.5)

ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim(fmin, fmax)

plt.tight_layout()
plt.show()
plt.close(fig)


# leftovers

# freflection_fft = fa_fft - forcing_fft_right * np.exp(-1j * 2 * np.pi * f * deltat_shift_right) # subtract forced wave
# greflection_fft = ga_fft - forcing_fft_left * np.exp(-1j * 2 * np.pi * f * deltat_shift_left)

# freflection = np.fft.ifft(freflection_fft, n=N).real
# greflection = np.fft.ifft(greflection_fft, n=N).real

freflection = fa[:]
greflection = ga[:]

# freflection[time < tmin_reflection] = 0.0  # remove initial signal, only retain the reflection
# greflection[time < tmin_reflection] = 0.0

# 1. Find indices corresponding to the reflection window
mask = (time >= tmin_reflection) & (time <= tmax_reflection)

# 2. Build a Tukey window of the correct length
window = scipy.signal.tukey(mask.sum(), alpha=0.1)

# 3. Create full-length zero arrays
window_full = np.zeros_like(time, dtype=float)

# 4. Insert the window into the correct region
window_full[mask] = window

# 5. Apply the window to signals
freflection = freflection * window_full
greflection = greflection * window_full

freflection_fft = np.fft.fft(freflection)[0:N//2]
greflection_fft = np.fft.fft(greflection)[0:N//2]

fin = fa[:]
gin = ga[:]

# freflection[time < tmin_reflection] = 0.0  # remove initial signal, only retain the reflection
# greflection[time < tmin_reflection] = 0.0

# 1. Find indices corresponding to the reflection window
maskin = (time <= tmin_reflection) & (time >= 0.0615)

# 2. Build a Tukey window of the correct length
window = scipy.signal.tukey(maskin.sum(), alpha=0.1)

# 3. Create full-length zero arrays
window_full = np.zeros_like(time, dtype=float)

# 4. Insert the window into the correct region
window_full[maskin] = window

# 5. Apply the window to signals
fin = fin * window_full
gin = gin * window_full

fin_fft = np.fft.fft(fin)[0:N//2]
gin_fft = np.fft.fft(gin)[0:N//2]

fig, ax = plt.subplots(figsize=(4,3))

ax.plot(time, freflection, label='$f_r$', color='r')
ax.plot(time, greflection, label='$g_r$', color='b')
ax.plot(time, fin, label='$f_in$', color='b', linestyle='dashed')
ax.plot(time, gin, label='$g_in$', color='r', linestyle='dashed')

    # ax.plot(timeout, uout, label='Signal Out', color='b')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.legend()
ax.grid(visible=True, which='major', color='k', linestyle='-')
ax.grid(visible=True, which='minor', color='k', linestyle='--', alpha=0.5)


plt.tight_layout()
plt.show()
plt.close(fig)

fig, ax = plt.subplots(figsize=(4,3))

ax.plot(f, np.abs(freflection_fft), label='$f_r$', color='r')
ax.plot(f, np.abs(forcing_fft_left), label='Outlet forcing', color='k', linestyle='dashed', alpha=0.5)

ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Magnitude')
ax.legend()
ax.grid('both')
ax.set_xlim(0, 1.5 * fmax)


plt.tight_layout()
plt.show()
plt.close(fig)

fig, ax = plt.subplots(figsize=(4,3))

ax.plot(f, np.abs(greflection_fft), label='$g_r$', color='b')
ax.plot(f, np.abs(forcing_fft_right), label='Inlet forcing', color='k', linestyle='dashed', alpha=0.5)

ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Magnitude')
ax.legend()
ax.grid('both')
ax.set_xlim(0, 1.5 * fmax)


plt.tight_layout()
plt.show()
plt.close(fig)

# reflection_coeff_inlet = freflection_fft / forcing_fft_left
# reflection_coeff_outlet = greflection_fft / forcing_fft_right

reflection_coeff_inlet = freflection_fft / gin_fft
reflection_coeff_outlet = greflection_fft / fin_fft

# reflection_coeff_inlet, f, h = wiener_hopf(shift(freflection, -deltat_shift_left//deltat), forcing_left, time, L=int(1.0 / c // deltat), deltat=deltat, plot=False, eps_conditioning=1e-10)
# reflection_coeff_outlet, f, h = wiener_hopf(shift(greflection, -deltat_shift_right//deltat), forcing_right, time, L=int(1.0 / c // deltat), deltat=deltat, plot=False, eps_conditioning=1e-10)


fig, ax = plt.subplots(figsize=(4,3))

ax.plot(f, np.abs(reflection_coeff_inlet), label=r'$\hat{r}_{inlet}$', color='r', alpha=0.5)
ax.plot(f, np.abs(reflection_coeff_outlet), label=r'$\hat{r}_{outlet}$', color='b', alpha=0.5)

args = np.where((f > fmin) & (f < fmax_interp))[0]
(K_IN, AIN), vfitIN, popt, pcov = fit_tauIO(f[args], np.abs(reflection_coeff_inlet)[args], Amax=2.0 * np.max(np.abs(reflection_coeff_inlet)[f < fmax_interp]))
print(f'sigma INLET: {K_IN / c:.8f}, A INLET = {AIN:.8f}')

ax.plot(f[args], vfitIN, color='r', linestyle='dashed', linewidth=3, label=rf'$\frac{{-1}}{{1+i \omega \tau}}$' + f', $\\sigma = {K_IN / c:.8f}$')

(K_OUT, AOUT), vfitOUT, popt, pcov = fit_tauIO(f[args], np.abs(reflection_coeff_outlet)[args], Amax=2.0 * np.max(np.abs(reflection_coeff_outlet)[f < fmax_interp]))
print(f'sigma OUTLET: {K_OUT / c:.8f}, A OUTLET = {AOUT:.8f}')

ax.plot(f[args], vfitOUT, color='b', linestyle='dashed', linewidth=3, label=rf'$\frac{{-1}}{{1+i \omega \tau}}$' + f', $\\sigma = {K_OUT / c:.8f}$')


ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Gain')

ax.legend()
ax.grid(visible=True, which='major', color='k', linestyle='-')
ax.grid(visible=True, which='minor', color='k', linestyle='--', alpha=0.5)

ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim(fmin, fmax)
ax.set_aspect('equal')

plt.tight_layout()
plt.show()
plt.close(fig)

