import tkinter as Tk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

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
import umap

# for display
import sys
import os
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using :0.0')
    os.environ.__setitem__('DISPLAY', ':0.0')

colormap = np.array([
    [0, 127, 70],
    [255, 0, 0],
    [255, 217, 38],
    [0, 135, 255],
    [165, 0, 165],
    [255, 167, 255],
    [97, 142, 151],
    [0, 255, 255],
    [255, 96, 38],
    [142, 76, 0],
    [33, 0, 127],
    [0, 0, 0],
    [183, 183, 183],
    [76, 255, 0],
], dtype=np.float) / 255 

class UI_encoder:
    def __init__(self):
        self.root = Tk.Tk()
        self.root.wm_title("Encoder Model Toolbox")
        self.embed_list = []
        self.filename_list = []
        self.speaker_list = []
        load_model(Path('./encoder/saved_models/my_run.pt'))
        self.speaker_name = Tk.StringVar()
        self.speaker_name.set('user01')
        self.entry_speaker = Tk.Entry(self.root, 
                                        textvariable = self.speaker_name,
                                        width=10)
        self.label_speaker = Tk.Label(self.root, text = 'Speaker Name')
        self.button_rec = Tk.Button(self.root, 
                                    bg = 'white', 
                                    activebackground='red',
                                    text = 'record', 
                                    command = self.record_wav)
        self.button_load = Tk.Button(self.root, bg = 'white', text = 'load file', command = self.load_wav)
        self.umap_hot = False

        # matplotlib canvas
        self.fig_org_wav,self.ax_org_wav = plt.subplots(2,1,figsize=(7,3))
        #self.fig_org_wav.suptitle('Input Wav file')
        self.ax_org_wav[0].set_title('Input Wav file')
        self.canv_org_wav = FigureCanvasTkAgg(self.fig_org_wav, master=self.root)
        self.canv_org_wav.draw()
        
        self.fig_prc_wav,self.ax_prc_wav = plt.subplots(2,1,figsize=(7,3))
        #self.fig_prc_wav.suptitle('After preprocessing')
        self.ax_prc_wav[0].set_title('After preprocessing')
        self.canv_prc_wav = FigureCanvasTkAgg(self.fig_prc_wav, master=self.root)
        self.canv_prc_wav.draw()

        self.fig_embed = plt.figure(figsize=(4,4))
        self.fig_embed.suptitle('Embedding')
        self.canv_embed = FigureCanvasTkAgg(self.fig_embed, master=self.root)

        self.fig_umap,self.ax_umap = plt.subplots(figsize=(4,4))
        self.fig_umap.suptitle('UMAP')
        self.canv_umap = FigureCanvasTkAgg(self.fig_umap, master=self.root)

        # packing:
        # self.label_speaker.grid(row=1,column=1)
        # self.entry_speaker.grid(row=2,column=1)
        # self.button_rec.grid(row=1,column=2)
        # self.button_load.grid(row=1,column=2)

        self.canv_org_wav.get_tk_widget().pack(side=Tk.TOP, fill=Tk.X, expand=1)
        self.canv_prc_wav.get_tk_widget().pack(side=Tk.TOP, fill=Tk.X, expand=1)
        self.canv_embed.get_tk_widget().pack(side=Tk.LEFT, expand=1)
        self.canv_umap.get_tk_widget().pack(side=Tk.LEFT, expand=1)
        self.button_rec.pack(side=Tk.RIGHT)
        self.button_load.pack(side=Tk.RIGHT)
        self.entry_speaker.pack(side=Tk.RIGHT)
        self.label_speaker.pack(side=Tk.RIGHT)

        


    def sample_update(self, wf_str):
        _filename = wf_str.split('/')[-1]
        wav_orig,fs = librosa.load(str(wf_str), sr=16000)
        wav_orig = normalize_volume(wav_orig, -30, increase_only=True)
        mel_orig = wav_to_mel_spectrogram(wav_orig)
        wav = preprocess_wav(wf_str, source_sr=fs)
        mel = wav_to_mel_spectrogram(wav)
        # original signal plot
        self.ax_org_wav[0].clear()
        self.ax_org_wav[1].clear()
        self.ax_org_wav[0].set_title('Input Wav file: \'%s\''%_filename)
        t_orig = np.arange(len(wav_orig))/fs
        self.ax_org_wav[0].plot(t_orig,wav_orig)
        self.ax_org_wav[0].autoscale(enable=True, axis='x', tight=True)
        librosa.display.specshow(librosa.power_to_db(mel_orig.T,ref=np.max),ax=self.ax_org_wav[1],y_axis='mel',x_axis='time',sr=fs)
        self.canv_org_wav.draw()
        # processed signal plot
        self.ax_prc_wav[0].clear()
        self.ax_prc_wav[1].clear()
        self.ax_prc_wav[0].set_title('After Prepreprocessing')
        t = np.arange(len(wav))/fs
        self.ax_prc_wav[0].plot(t,wav)
        self.ax_prc_wav[0].autoscale(enable=True, axis='x', tight=True)
        librosa.display.specshow(librosa.power_to_db(mel.T,ref=np.max),ax=self.ax_prc_wav[1],y_axis='mel',x_axis='time',sr=fs)
        self.canv_prc_wav.draw()
        # Embedding plot
        embed = embed_utterance(wav)
        self.fig_embed.clear()
        self.fig_embed.suptitle('Embedding')
        ax_embed = self.fig_embed.add_subplot()
        plot_embedding_as_heatmap(embed, ax=ax_embed)
        self.canv_embed.draw()
        self.embed_list.append(embed)
        self.speaker_list.append(self.speaker_name.get())
        self.filename_list.append(_filename)
        self.plot_umap()

    def plot_umap(self):
        min_umap_points = 5
        embeds = self.embed_list
        self.ax_umap.clear()
        speakers = np.unique(self.speaker_list)
        colors = {speaker_name: colormap[i] for i, speaker_name in enumerate(speakers)}
        if len(self.embed_list) < min_umap_points:
            self.ax_umap.text(.5, .5, "Add %d more points to\ngenerate the projections" % 
                              (min_umap_points - len(self.embed_list)), 
                              horizontalalignment='center', fontsize=15)
        else:
            if not self.umap_hot:
                print("Drawing UMAP projections for the first time, this will take a few seconds.")
                self.umap_hot = True
            reducer = umap.UMAP(int(np.ceil(np.sqrt(len(embeds)))), metric="cosine")
            projections = reducer.fit_transform(embeds)
            speakers_done = set()
            for projection, speakername in zip(projections, self.speaker_list):
                color = colors[speakername]
                mark = "."
                label = None if speakername in speakers_done else speakername
                speakers_done.add(speakername)
                self.ax_umap.scatter(projection[0], projection[1], c=[color], marker=mark,
                                     label=label)
            self.ax_umap.legend(prop={'size': 10})
            self.ax_umap.set_aspect("equal", "datalim")
            self.ax_umap.set_xticks([])
            self.ax_umap.set_yticks([])
        self.canv_umap.draw()

    def record_wav(self):
        self.button_rec.config(bg = 'red')
        self.root.update()
        sample_format = pyaudio.paInt16
        channels = 1
        fs = 16000
        seconds = 8
        speaker = self.speaker_name.get()
        i = 0
        while (Path.exists(Path(speaker+'_'+str(i).zfill(2)+'.wav'))):
            i += 1
        filename = speaker+'_'+str(i).zfill(2)+'.wav'
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
        print('Done...File: %s'%filename)
        self.button_rec.config(bg = 'white')
        self.root.update()
        self.sample_update(filename)
    
    def load_wav(self):
        filename = Tk.filedialog.askopenfilename()
        self.sample_update(filename)


ui_ = UI_encoder()
ui_.root.mainloop()
