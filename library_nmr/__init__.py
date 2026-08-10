"""
library_nmr — solid-state NMR processing pipeline for battery electrolyte
materials (LLZO, LATP), built around Bruker TopSpin raw data and nmrglue.

Developed during a postdoc on solid electrolyte characterization
(⁶Li/⁷Li, ²⁷Al, ³¹P, ¹³⁹La), as a self-taught Python project.

Shared building blocks (core.py, fitting.py) are used across the
task-specific scripts (1D processing, T1/T2 relaxation, MQMAS, DFT/GIPAW
calibration) — see the README for an overview of each script.
"""

__version__ = "0.1.0"
