import matplotlib.pyplot as plt
from library_nmr.relaxation_T1_onepulse_series import process_1d_spectrum, DATASETS, LB, PH1, ZF_FACTOR, REFERENCE_SHIFT_PPM

path = DATASETS[150]

for ph0_test in [20, 22]:
    delta, spectrum, dic, ph0_used = process_1d_spectrum(
        path, LB, ph0_manual=ph0_test, ph1=PH1, zf_factor=ZF_FACTOR,
        auto_ph0=False, reference_shift_ppm=REFERENCE_SHIFT_PPM
    )
    plt.plot(delta, spectrum.real, label=f"PH0={ph0_test}°")

plt.xlim(30, -30)
plt.axhline(0, color="gray", lw=0.5)
plt.legend()
plt.show()