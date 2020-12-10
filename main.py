#%%
import pyaudio 
import wave
import struct
import numpy as np
from matplotlib import pyplot as plt
from encoder.inference import *
from encoder.audio import *
import librosa
import librosa.display
from pathlib import Path

# Recording
sample_format = pyaudio.paInt16
channels = 1
fs = 16000
seconds = 10
filename = "rec.wav"

p = pyaudio.PyAudio()
print('Recording...')
stream = p.open(format=sample_format,
                channels=channels,
                rate=fs,
                frames_per_buffer=fs,
                input=True)

rec_sound = stream.read(fs*seconds)
stream.stop_stream()
stream.close()
p.terminate()
wf = wave.open(filename, 'wb')
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(sample_format))
wf.setframerate(fs)
wf.writeframes(rec_sound)
wf.close
print('Done...')

# preprocessing
#%%
wav_orig,_ = librosa.load(str(filename), sr=None)
wav_orig = normalize_volume(wav_orig, -30, increase_only=True)
mel_orig = wav_to_mel_spectrogram(wav_orig)
wav = preprocess_wav(filename, source_sr=fs)
mel = wav_to_mel_spectrogram(wav)
fig,ax = plt.subplots(4,1)
t_orig = np.arange(0,seconds,1/fs)
ax[0].plot(t_orig,wav_orig)
img = librosa.display.specshow(librosa.power_to_db(mel_orig.T,ref=np.max),ax=ax[1],y_axis='mel',x_axis='time',sr=fs)
t = np.arange(len(wav))/fs
ax[2].plot(t,wav)
img = librosa.display.specshow(librosa.power_to_db(mel.T,ref=np.max),ax=ax[3],y_axis='mel',x_axis='time',sr=fs)
plt.show()

# Embedding
# %%
load_model(Path('./encoder/saved_models/my_run.pt'))
embed = embed_utterance(wav)
fig,ax = plt.subplots()
plot_embedding_as_heatmap(embed, ax=ax)
plt.show()
# %%
