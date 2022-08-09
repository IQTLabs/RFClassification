import os
import numpy as np 
import zstandard
from tqdm import tqdm
import torch
from scipy import signal
import matplotlib
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import torchvision
from torchvision import datasets, models, transforms

from gamutrf.sample_reader import get_reader
from gamutrf.utils import parse_filename 


class GamutRFDataset(torch.utils.data.Dataset): 
    def __init__(self, label_dirs, sample_secs=0.02, nfft=1024, transform=None):
        
        self.sample_secs = sample_secs
        self.nfft = nfft
        labeled_filenames = self.labeled_files(label_dirs)
        self.idx = self.idx_info(labeled_filenames, sample_secs)
 
        unique_labels = sorted(list(set(self.idx[:,0])))
        self.unique_labels = unique_labels
        self.class_to_idx = {lbl:i for i,lbl in enumerate(self.unique_labels)}
        self.idx_to_class = {i:lbl for i,lbl in enumerate(self.unique_labels)}
        
        self.cmap = plt.get_cmap('jet')
        
        self.transform = transform
        if self.transform is None: 
            self.transform = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Resize((256, 256))
            ])
        
    def __len__(self):
        return len(self.idx)
    
    def __getitem__(self, i):
        
        label_str = self.idx[i,0]
        filename = self.idx[i,1]
        byte_offset = self.idx[i,2]
        
        
        # parse filename and get samples 
        start = timer()
        freq_center, sample_rate, sample_dtype, sample_bytes, sample_type, sample_bits = parse_filename(filename)
        start = timer()
        samples = self.read_recording(filename, sample_rate, sample_dtype, sample_bytes, self.sample_secs, seek_bytes=byte_offset)
        #print(f"read_recording() time = {timer()-start}")
        # get spectrogram using scipy.signal.spectrogram()
        
        #win_type='hann'
        f, t, S = signal.spectrogram(samples, sample_rate, window=signal.hann(self.nfft, sym=False), nperseg=self.nfft, detrend='constant', return_onesided=False) 
        f = np.fft.fftshift(f)
        S = np.fft.fftshift(S, axes=0)
        S = 10 * np.log10(S) # dB scale transform
        
        # normalize spectrogram (0 to 1) 
        S_norm = (S-np.min(S))/(np.max(S) - np.min(S))
        
        rgba_img = self.cmap(S_norm)
        rgb_img = np.delete(rgba_img, 3, 2)
        
        #data = np.float32(np.moveaxis(rgb_img, -1, 0))
        data = self.transform(np.float32(rgb_img))
        label = torch.tensor(self.class_to_idx[label_str])
        return data, label, S
    
    def labeled_files(self, label_dirs): 
        labeled_filenames = {}
        for label, label_dir in label_dirs.items(): 
            label_dir = list(label_dir) if type(label_dir) is not list else label_dir
            valid_files = []
            for d in label_dir:
                valid_files.extend([d+filename for filename in os.listdir(d) if (filename.endswith('.zst') or filename.endswith('.raw')) and not filename.startswith('.')])

            labeled_filenames[label] = valid_files

            print(f"{label=}, {len(valid_files)} files")

        return labeled_filenames

    def idx_info(self, labeled_filenames, sample_secs = 0.02): 
        idx = []

        for label, valid_files in labeled_filenames.items(): 
            for i,full_filename in enumerate(tqdm(valid_files)): 
                idx_filename = full_filename+f"_{str(sample_secs)}.npy"
                if os.path.exists(idx_filename): 
                    start = timer()
                    file_idx = np.load(idx_filename).tolist()
                    #print(f"loading {idx_filename}; {i}/{len(valid_files)} time = {timer()-start}")
                else: 
                    file_idx = []
                    freq_center, sample_rate, sample_dtype, sample_len, sample_type, sample_bits = parse_filename(full_filename)
                    start = timer()
                    infile = zstandard.ZstdDecompressor().stream_reader(open(full_filename, 'rb'))
                    while True: 
                        start_byte = infile.tell()
                        sample_buffer = infile.read(int(sample_rate * sample_secs) * sample_len)
                        buffered_samples = int(len(sample_buffer) / sample_len)
                        if buffered_samples != int(sample_rate*sample_secs):
                            break
                        file_idx.append([label, full_filename, start_byte])
                    np.save(idx_filename, file_idx)
                    #print(f"saving {idx_filename}; {i}/{len(valid_files)} time = {timer()-start}")
                idx.extend(file_idx)
        return np.array(idx)
    
    def debug(self, i):
        
        label_str = self.idx[i,0]
        filename = self.idx[i,1]
        byte_offset = self.idx[i,2]
        
        # parse filename and get samples 
        
        
        start = timer()
        freq_center, sample_rate, sample_dtype, sample_bytes, sample_type, sample_bits = parse_filename(filename)
        samples2 = next(self.read_recording2(filename, sample_rate, sample_dtype, sample_bytes, sample_secs=self.sample_secs, skip_sample_secs=int(byte_offset)/sample_bytes/sample_rate, max_sample_secs=self.sample_secs))
        print('get samples2 time = ',timer()-start)
        
        start = timer()
        freq_center, sample_rate, sample_dtype, sample_bytes, sample_type, sample_bits = parse_filename(filename)
        samples = self.read_recording(filename, sample_rate, sample_dtype, sample_bytes, self.sample_secs, seek_bytes=byte_offset)
        print('get samples time = ',timer()-start)
        
        print(f"{samples.shape=}")
        print(f"{samples2.shape=}")
        n_per_seg=self.nfft
        win_type='hann'
        
        # get spectrogram using plt.specgram()
        fig, ax = plt.subplots(1, 3, figsize=(15,3), dpi=100)
        start = timer()
        Pxx, freqs, bins, im = ax[0].specgram(samples, NFFT=n_per_seg, Fs=sample_rate)
        Pxx = 10 * np.log10(Pxx)
        print('plt.specgram() time = ',timer()-start)
        plt.colorbar(im, ax=ax[0])
        ax[0].set_title('plt.specgram()')
        im = ax[1].pcolormesh(bins, freqs, Pxx)
        ax[1].set_title('plt.specgram() pcolormesh()')
        plt.colorbar(im, ax=ax[1])
        im = ax[2].imshow(Pxx, origin='lower', aspect='auto')
        ax[2].set_title('plt.specgram() imshow()')
        plt.colorbar(im, ax=ax[2])
        plt.show()
        print(Pxx.shape)
        
        # get spectrogram using plt.specgram()
        fig, ax = plt.subplots(1, 3, figsize=(15,3), dpi=100)
        start = timer()
        Pxx, freqs, bins, im = ax[0].specgram(samples2, NFFT=n_per_seg, Fs=sample_rate)
        Pxx = 10 * np.log10(Pxx)
        print('plt.specgram() time = ',timer()-start)
        plt.colorbar(im, ax=ax[0])
        ax[0].set_title('plt.specgram()')
        im = ax[1].pcolormesh(bins, freqs, Pxx)
        ax[1].set_title('plt.specgram() pcolormesh()')
        plt.colorbar(im, ax=ax[1])
        im = ax[2].imshow(Pxx, origin='lower', aspect='auto')
        ax[2].set_title('plt.specgram() imshow()')
        plt.colorbar(im, ax=ax[2])
        plt.show()
        print(Pxx.shape)
        
        #######
        
        # get spectrogram using scipy.signal.spectrogram()
        fig, ax = plt.subplots(2, 3, figsize=(20,8), dpi=100)
        fig.suptitle('signal.spectrogram() window=signal.hann(n_per_seg, sym=True)')
        plt.tight_layout()
        start = timer()
        #f, t, S = signal.spectrogram(samples, sample_rate, window = plt.mlab.window_hanning(np.ones(n_per_seg)), nperseg=n_per_seg, detrend=False, return_onesided=False) 
        f, t, S = signal.spectrogram(samples, sample_rate, window=signal.hann(n_per_seg, sym=True), nperseg=n_per_seg, detrend='constant', return_onesided=False) 
        f = np.fft.fftshift(f)
        S = np.fft.fftshift(S, axes=0)
        S = 10 * np.log10(S) # dB scale transform
        print(S.shape)
        print('signal.spectrogram() time = ',timer()-start)
        
        # plot spectrogram 
        im = ax[0,1].pcolormesh(t, f, S)
        plt.colorbar(im, ax=ax[0,1])
        ax[0,1].set_title('signal.spectrogram() pcolormesh()')
        
        # plot spectrogram imshow
        im = ax[0,2].imshow(S, aspect='auto', origin='lower')
        plt.colorbar(im, ax=ax[0,2])
        ax[0,2].set_title('signal.spectrogram() imshow()')
        
        # standardize spectrogram and plot 
        # standardization (mean 0, std dev 1)
        S_std = (S-np.mean(S))/(np.sqrt(np.var(S)))
        im = ax[1,0].imshow(S_std, aspect='auto', origin='lower')
        plt.colorbar(im, ax=ax[1,0])
        ax[1,0].set_title('signal.spectrogram() standardized imshow()')

        # normalize spectrogram and plot
        # normalization (0 to 1) 
        S_norm = (S-np.min(S))/(np.max(S) - np.min(S))
        im = ax[1,1].imshow(S_norm, aspect='auto', origin='lower')
        plt.colorbar(im, ax=ax[1,1])
        ax[1,1].set_title('signal.spectrogram() normalized imshow()')
        
        # normalize spectrogram and plot interpolation=nearest
        # normalization (0 to 1) 
        S_norm = (S-np.min(S))/(np.max(S) - np.min(S))
        im = ax[1,2].imshow(S_norm, aspect='auto', origin='lower', interpolation='nearest')
        plt.colorbar(im, ax=ax[1,2])
        ax[1,2].set_title('signal.spectrogram() normalized imshow( interpolation=nearest )')
        plt.show()
        
        # get PSD using scipy.signal.welch()
        start=timer()
        f_psd, P_psd = signal.welch(samples, sample_rate, window=win_type, nperseg=n_per_seg, return_onesided=False, detrend=False)
        
        #print(P_psd)
#         print(f"{P_psd=}")
#         #print(f"{np.fft.fftshift(P_psd)=}")
#         print(f"{np.max(P_psd)=}")
#         print(f"{np.mean(P_psd)=}")
        print('psd time = ',timer()-start)
        fig, ax = plt.subplots(1, 2, figsize=(10,3), dpi=100)
        im = ax[0].plot(np.fft.fftshift(f_psd), np.fft.fftshift(P_psd))
        ax[0].set_title('signal.welch() detrend=False')
        
        # get PSD using scipy.signal.welch()
        start=timer()
        f_psd, P_psd_detrend = signal.welch(samples, sample_rate, window=win_type, nperseg=n_per_seg, return_onesided=False)
#         print(f"{P_psd_detrend=}")
#         #print(f"{np.fft.fftshift(P_psd)=}")
#         print(f"{np.max(P_psd_detrend)=}")
#         print(f"{np.mean(P_psd_detrend)=}")
#         print(P_psd - P_psd_detrend)
#         print('psd time = ',timer()-start)
        im = ax[0].plot(np.fft.fftshift(f_psd), np.fft.fftshift(P_psd_detrend))
        ax[1].set_title('signal.welch()')
        plt.show()

        #######
        # get spectrogram using scipy.signal.spectrogram()
        fig, ax = plt.subplots(2, 3, figsize=(20,8), dpi=100)
        fig.suptitle('signal.spectrogram() window=signal.hann(n_per_seg, sym=False)')
        plt.tight_layout()
        start = timer()
        f, t, S = signal.spectrogram(samples, sample_rate, window=signal.hann(n_per_seg, sym=False), nperseg=n_per_seg, detrend=False, return_onesided=False) 
        f = np.fft.fftshift(f)
        S = np.fft.fftshift(S, axes=0)
        S = 10 * np.log10(S) # dB scale transform
        print(S.shape)
        print('signal.spectrogram() time = ',timer()-start)
        
        # plot spectrogram 
        im = ax[0,1].pcolormesh(t, f, S)
        plt.colorbar(im, ax=ax[0,1])
        ax[0,1].set_title('signal.spectrogram() pcolormesh()')
        
        # plot spectrogram imshow
        im = ax[0,2].imshow(S, aspect='auto', origin='lower')
        plt.colorbar(im, ax=ax[0,2])
        ax[0,2].set_title('signal.spectrogram() imshow()')
        
        # standardize spectrogram and plot 
        # standardization (mean 0, std dev 1)
        S_std = (S-np.mean(S))/(np.sqrt(np.var(S)))
        im = ax[1,0].imshow(S_std, aspect='auto', origin='lower')
        plt.colorbar(im, ax=ax[1,0])
        ax[1,0].set_title('signal.spectrogram() standardized imshow()')

        # normalize spectrogram and plot
        # normalization (0 to 1) 
        S_norm = (S-np.min(S))/(np.max(S) - np.min(S))
        im = ax[1,1].imshow(S_norm, aspect='auto', origin='lower')
        plt.colorbar(im, ax=ax[1,1])
        ax[1,1].set_title('signal.spectrogram() normalized imshow()')
        
        # normalize spectrogram and plot interpolation=nearest
        # normalization (0 to 1) 
        S_norm = (S-np.min(S))/(np.max(S) - np.min(S))
        im = ax[1,2].imshow(S_norm, aspect='auto', origin='lower', interpolation='nearest')
        plt.colorbar(im, ax=ax[1,2])
        ax[1,2].set_title('signal.spectrogram() normalized imshow( interpolation=nearest )')
        plt.show()

        # get PSD using scipy.signal.welch()
        start=timer()
        f_psd, P_psd = signal.welch(samples, sample_rate, window=win_type, nperseg=n_per_seg, return_onesided=False, detrend=False)
        print('psd time = ',timer()-start)
        fig, ax = plt.subplots(1, 2, figsize=(10,3), dpi=100)
        im = ax[0].plot(np.fft.fftshift(f_psd), np.fft.fftshift(P_psd))
        ax[0].set_title('signal.welch() detrend=False')
        
        # get PSD using scipy.signal.welch()
        start=timer()
        f_psd, P_psd = signal.welch(samples, sample_rate, window=win_type, nperseg=n_per_seg, return_onesided=False)
        print('psd time = ',timer()-start)
        im = ax[1].plot(np.fft.fftshift(f_psd), np.fft.fftshift(P_psd))
        ax[1].set_title('signal.welch()')
        plt.show()
        
        
        #######
        
        label = self.class_to_idx[label_str]
        
        return samples, label
    
    def read_recording2(self, filename, sample_rate, sample_dtype, sample_len, sample_secs=1.0, skip_sample_secs=0, max_sample_secs=0):
        """Read an I/Q recording and iterate over it, returning 1-D numpy arrays of csingles, of size sample_rate * sample_secs.
        Args:
            filename: str, recording to read.
            sample_rate: int, samples per second
            sample_dtype: numpy.dtype, binary format of original I/Q recording.
            sample_len: int, length of one sample.
            sample_secs: float, number of seconds worth of samples per iteration.
            skip_sample_secs: float, number of seconds worth of samples to skip initially.
            max_sample_secs: float, maximum number of seconds of samples to read (or None for all).
        Returns:
            numpy arrays of csingles.
        """
        reader = get_reader(filename)
        samples_remaining = 0
        if max_sample_secs:
            samples_remaining = int(sample_rate * max_sample_secs)
        with reader(filename) as infile:
            if skip_sample_secs:
                start = timer()
                infile.read(int(sample_rate * skip_sample_secs) * sample_len)
                print(f"read skip = {timer()-start} seconds")
            while True:
                if max_sample_secs and not samples_remaining:
                    break
                start = timer() 
                sample_buffer = infile.read(int(sample_rate * sample_secs) * sample_len)
                print(f"read = {timer()-start} seconds")
                buffered_samples = int(len(sample_buffer) / sample_len)
                if buffered_samples == 0:
                    break
                if max_sample_secs:
                    if buffered_samples <= samples_remaining:
                        samples_remaining -= buffered_samples
                    else:
                        buffered_samples = samples_remaining
                        samples_remaining = 0
                x1d = np.frombuffer(sample_buffer, dtype=sample_dtype, count=buffered_samples)
                yield x1d['i'] + np.csingle(1j) * x1d['q']
            
    def read_recording(self, filename, sample_rate, sample_dtype, sample_bytes, sample_secs=1.0, seek_bytes=0):
        """Read an I/Q recording and iterate over it, returning 1-D numpy arrays of csingles, of size sample_rate * sample_secs.
        Args:
            filename: str, recording to read.
            sample_rate: int, samples per second
            sample_dtype: numpy.dtype, binary format of original I/Q recording.
            sample_len: int, length of one sample.
            sample_secs: float, number of seconds worth of samples per iteration.
        Returns:
            numpy arrays of csingles.
        """

        reader = get_reader(filename)
#         with reader(filename) as infile:
#             start = timer() 
#             infile.seek(int(seek_bytes))
#             print(f"seek = {timer()-start} seconds")
            
#         with reader(filename) as infile:
#             start = timer()
#             infile.read(int(seek_bytes))
#             print(f"read seek = {timer()-start} seconds")
           
#         avg_time = 0
#         avg_len = 20 
#         for i in range(avg_len): 
#             with reader(filename) as infile:
#                 start = timer() 
#                 infile.seek(int(seek_bytes))
#                 duration = timer()-start
#                 avg_time += duration 
#                 print(f"seek = {duration} seconds")
#         avg_time /= avg_len
#         print(f"average seek time = {avg_time} seconds")
#         avg_time = 0
#         avg_len = 20 
#         for i in range(avg_len): 
#             with reader(filename) as infile:
#                 start = timer() 
#                 infile.read(int(seek_bytes))
#                 duration = timer()-start
#                 avg_time += duration
#                 print(f"read = {duration} seconds")
#         avg_time /= avg_len
#         print(f"average read time = {avg_time} seconds")
        
        with reader(filename) as infile:
            start = timer() 
            infile.seek(int(seek_bytes))
            #print(f"seek = {timer()-start} seconds")
            
            start = timer() 
            sample_buffer = infile.read(int(sample_rate * sample_secs) * sample_bytes)
            #print(f"read = {timer()-start} seconds")
            buffered_samples = int(len(sample_buffer) / sample_bytes)
            if buffered_samples == 0:
                raise 
            x1d = np.frombuffer(sample_buffer, dtype=sample_dtype,
                                count=buffered_samples)
            return x1d['i'] + np.csingle(1j) * x1d['q']