import matplotlib.pyplot as plt
from library_nmr.relaxation_T1_onepulse_series import (process_1d_spectrum, DATASETS, LB, PH0_MANUAL, PH1, ZF_FACTOR, AUTO_PH0, REFERENCE_SHIFT_PPM)

for d1 in [0.05, 1, 10, 150]:  # quelques points bien espacés
    path = DATASETS[d1]
    delta, spectrum, dic, ph0_used = process_1d_spectrum(
        path, LB, PH0_MANUAL, PH1, ZF_FACTOR,
        auto_ph0=AUTO_PH0, reference_shift_ppm=REFERENCE_SHIFT_PPM
    )
    plt.plot(delta, spectrum.real, label=f"D1={d1}s (PH0={ph0_used:.1f}°)")

plt.xlim(30, -30)
plt.legend()
plt.show()