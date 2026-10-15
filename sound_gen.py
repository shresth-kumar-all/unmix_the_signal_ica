import numpy as np
from scipy.io import wavfile as wav

fs = 44100  # Sample rate (CD quality)
duration = 3.0  # 3 seconds long
t = np.linspace(0, duration, int(fs * duration))

# Create 3 distinct, non-Gaussian sound waves (tones/chirps)
# Sound 0: A clean 440 Hz tone (A note)
sound_0 = np.sin(2 * np.pi * 440 * t)

# Sound 1: A square-like wave (sharper, non-Gaussian harmonics)
sound_1 = np.sign(np.sin(2 * np.pi * 220 * t)) * 0.5

# Sound 2: A higher 880 Hz tone with a changing frequency (chirp)
sound_2 = np.sin(2 * np.pi * 880 * t + np.sin(t * 5))

# Save them as the files your script expects
wav.write("Sources\\Sound_0.wav", fs, sound_0.astype(np.float32))
wav.write("Sources\\Sound_1.wav", fs, sound_1.astype(np.float32))
wav.write("Sources\\Sound_2.wav", fs, sound_2.astype(np.float32))

print("Test sound files generated successfully!")