import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

fpr = 16000
T = 0.1
f = 500
name = "Filip"

binary=''.join(format(ord(i),"08b") for i in name)
#print(binary)

t_bit = np.linspace(0, T, int(T * fpr), endpoint=False)

signals= []
for bit in binary:
    if bit == '0':
        signals.append(np.sin(2*np.pi*f*t_bit))
    else:
        signals.append(-np.sin(2*np.pi*f*t_bit))
signals = np.concatenate(signals)

#print(signals)

total_time = np.linspace(0, len(binary) * T,len(signals), endpoint=False)

for sample in [8000,16000,24000,32000,48000]:
    print(f"Playing {sample} khz samples")
    #sd.play(signals, sample)
    sd.wait()


plt.figure(figsize=(20, 10))
plt.plot(total_time, signals, label="Signals", linewidth=3)
#plt.xlim(0, 0.01)
plt.grid(True)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

## QPSK

bit_pairs={
    "00" :np.sin(2*np.pi*f*t_bit),
    "01" :np.sin(2*np.pi*f*t_bit + np.pi/2),
    "10" :np.sin(2*np.pi*f*t_bit + np.pi),
    "11" :np.sin(2*np.pi*f*t_bit + 3*np.pi/2),
}
bit_groups = [binary[i:i+2] for i in range(0, len(binary), 2)]
print(bit_groups)

signals_QPSK = [bit_pairs[bits] for bits in bit_groups]
signals_QPSK = np.concatenate(signals_QPSK)

qpsk_time=np.linspace(0, len(bit_groups) * T, len(signals_QPSK), endpoint=False)

plt.plot(qpsk_time, signals_QPSK, label="Signals", linewidth=3)
plt.xlim(0,0.01)
plt.show()

for sample in [8000,16000,24000,32000,48000]:
    print(f"Playing {sample} khz samples")
    #sd.play(signals_QPSK, sample)
    sd.wait()

