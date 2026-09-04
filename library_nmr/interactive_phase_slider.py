import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import nmrglue as ng

from library_nmr.core import find_grpdly_shift

PATH = r"D:\Postdoc\Datas\LLZO-400-aug26\236"
LB = 10
ZF_FACTOR = 1
REFERENCE_SHIFT_PPM = 2
PH0_INIT = 21
PH1_INIT = -49.524

dic, data = ng.bruker.read(PATH, read_procs=False)
grpdly_shift = find_grpdly_shift(dic)
if grpdly_shift > 0:
    data = np.roll(data, -grpdly_shift)

N = data.shape[0]
dt = 1 / dic["acqus"]["SW_h"]
data_zf = np.concatenate([data, np.zeros(ZF_FACTOR * N, dtype=complex)])

t = np.arange(len(data_zf)) * dt
w = np.exp(-t * LB)
fid_apodized = w * data_zf
raw_spectrum = np.fft.fftshift(np.fft.fft(fid_apodized))  # pas encore phase

n = np.arange(len(raw_spectrum))
Nfft = len(raw_spectrum)

f = np.fft.fftshift(np.fft.fftfreq(len(data_zf), dt))
delta = (dic["acqus"]["O1"] - f) / dic["acqus"]["SFO1"]
delta = delta + REFERENCE_SHIFT_PPM


def phased(ph0_deg, ph1_deg):
    pivot = Nfft / 2  # centre du spectre (~0 ppm, pres du pic d'interet) -- PH1 ne bouge plus ce point
    phase = np.deg2rad(ph0_deg) + np.deg2rad(ph1_deg) * ((n - pivot) / Nfft)
    return (raw_spectrum * np.exp(1j * phase)).real


fig, (ax_full, ax_side) = plt.subplots(2, 1, figsize=(10, 8))
plt.subplots_adjust(bottom=0.25)

line_full, = ax_full.plot(delta, phased(PH0_INIT, PH1_INIT), color="blue")
ax_full.axhline(0, color="gray", lw=0.8)
ax_full.set_xlim(400, -400)
ax_full.set_title("Spectre complet")

line_side, = ax_side.plot(delta, phased(PH0_INIT, PH1_INIT), color="red")
ax_side.axhline(0, color="gray", lw=0.8)
ax_side.set_xlim(150, 90)  # zoome sur la bande ~+120 ppm -- change ces bornes si besoin
ax_side.set_title("Zoom bande de rotation")

ax_ph0 = plt.axes([0.15, 0.12, 0.7, 0.03])
ax_ph1 = plt.axes([0.15, 0.06, 0.7, 0.03])
slider_ph0 = Slider(ax_ph0, "PH0", -180, 180, valinit=PH0_INIT)
slider_ph1 = Slider(ax_ph1, "PH1", -300, 300, valinit=PH1_INIT)


def update(val):
    y = phased(slider_ph0.val, slider_ph1.val)
    line_full.set_ydata(y)
    line_side.set_ydata(y)
    ax_full.relim(); ax_full.autoscale_view(scalex=False)
    ax_side.relim(); ax_side.autoscale_view(scalex=False)
    fig.canvas.draw_idle()


slider_ph0.on_changed(update)
slider_ph1.on_changed(update)

plt.show()

print(f"\nValeurs finales : PH0 = {slider_ph0.val:.3f} deg, PH1 = {slider_ph1.val:.3f} deg")