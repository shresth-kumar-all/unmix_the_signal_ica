import numpy as np
from scipy.io import wavfile as wav
import random

#############################################################################################################################################
#sound loading:

sounds = 3 #Number of sources

sources = [] #Has the soundfiles
sfreq = [] #Has the sample frequencies

for c0 in range(0, sounds):
    tosend1 = wav.read(f"Sources\\Sound_{c0}.wav")[0]
    tosend2 = wav.read(f"Sources\\Sound_{c0}.wav")[1]
    sfreq.append(tosend1)
    if len(tosend2.shape) > 1: # Fixed shape check for multi-channel audio
        tosend2 = tosend2[:,0]
    sources.append(tosend2)

durations = [len(s) for s in sources]
min_dur = min(durations)

for c2 in range(0, sounds):
    sources[c2] = sources[c2][:min_dur] #all sounds are now of the same duration

S = np.array(sources, dtype=float) #matrix of amplitudes

#Randomizer matrix:
random_nums = []
for c3 in range(0, sounds):
    tosend = [random.random() for _ in range(sounds)]
    random_nums.append(tosend)

Rd = np.array(random_nums, dtype=float)
samples = Rd.dot(S) #Rd x S matrix multiplication (Mixed signals)

#############################################################################################################################################
#centering mixed signals:

avg_2d = []
for c4 in range(0, sounds):
    pro = samples[c4]
    n = len(pro)
    tot = sum(pro)
    mean = tot / n
    avg_2d.append([mean] * n)

avg = np.array(avg_2d)
CentSamps = samples - avg

#############################################################################################################################################
#PCA whitening:

covarience = np.cov(CentSamps)
evals, evectors = np.linalg.eigh(covarience)

#converting the tilted oval cloud into an upright, symmetric sphere with epsilon to prevent division by zero
epsilon = 1e-5
A = evectors.T
upright_samps = A.dot(CentSamps)

for c5 in range(0, sounds):
    pro = upright_samps[c5]
    tosend = pro / np.sqrt(evals[c5] + epsilon) # Added epsilon here!
    upright_samps[c5] = tosend

#############################################################################################################################################
#FastICA:

mem = np.zeros((sounds, sounds))
for c6 in range(0, sounds):
    r = np.random.randn(sounds)
    mag = (sum(k**2 for k in r))**(1/2)
    r = r / (mag + epsilon) # Added epsilon here just in case

    idkbro = 1000 #number of iterations for convergence
    for c61 in range(0, idkbro):
        r_prev = r.copy()
        proj = np.dot(r, upright_samps)
        tanh_proj = np.tanh(proj) #Non-Gaussianity contrast function
        d_tanh = 1 - tanh_proj**2 #derivative of tanh

        #fixed-point update rule (climbs the spikiness hill)
        r = np.mean(upright_samps * tanh_proj, axis=1) - np.mean(d_tanh) * r
        mag = (sum(k**2 for k in r))**(1/2)
        r = r / (mag + epsilon) # Added epsilon here!

    #Gram-Schmidt decorrelation loop to prevent finding the same component twice
    for c62 in range(0, c6):
        r = r - np.dot(r, mem[c62]) * mem[c62]
        mag = (sum(k**2 for k in r))**(1/2)
        r = r / (mag + epsilon) # Added epsilon here!

    mem[c6] = r #storing found unmixing vectors in memory

S_prob = np.dot(mem, upright_samps)

#############################################################################################################################################
#saving estimated sounds:

for i in range(sounds):
    out = S_prob[i]
    max_val = np.max(np.abs(out))
    if max_val > epsilon:
        out = out / max_val #normalize safely with epsilon check
    wav.write(f"Found\\Found_Sounds_{i}.wav", sfreq[0], out.astype(np.float32))

#############################################################################################################################################
#Evaluation

correlation_matrix = np.corrcoef(S, S_prob)
cross_corr = correlation_matrix[:sounds, sounds:]

print("\n--- Evaluation Results ---")
print("Cross-Correlation Matrix (Rows = Originals, Cols = Estimated):")
print(np.round(cross_corr, 3))

for i in range(sounds):
    best_match_idx = np.argmax(np.abs(cross_corr[i]))
    best_corr = cross_corr[i, best_match_idx]
    print(f"Original Source {i} best matches Estimated Source {best_match_idx} with a correlation of {best_corr:.3f}")