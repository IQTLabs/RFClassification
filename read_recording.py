# from Gamut repo
# collection of functions and scrips that reads rf recordings
import os
import re
import numpy as np
import zstandard

def read_recording(filename, sample_rate, sample_dtype, sample_len):
    """Read an I/Q recording and iterate over it, returning 1-D numpy arrays of csingles, of size sample_len.
    Args:
        filename: str, recording to read.
        sample_rate: int, samples per second
        sample_dtype: numpy.dtype, binary format of original I/Q recording.
        sample_len: int, samples per second.
    Returns:
        numpy arrays of csingles.
    """
    reader = get_reader(filename)
    with reader(filename) as infile:
        while True:
            sample_buffer = infile.read(sample_rate * sample_len)
            buffered_samples = int(len(sample_buffer) / sample_len)
            if buffered_samples == 0:
                break
            x1d = np.frombuffer(sample_buffer, dtype=sample_dtype,
                                count=buffered_samples)
            yield x1d['i'] + np.csingle(1j) * x1d['q']
            

SAMPLE_FILENAME_RE = re.compile(r'^.+_([0-9]+)Hz_([0-9]+)sps\.(s\d+|raw).*$')

def is_fft(filename):
    return os.path.basename(filename).startswith('fft_')

SAMPLE_DTYPES = {
    's8':  ('<i1', 'signed-integer'),
    's16': ('<i2', 'signed-integer'),
    's32': ('<i4', 'signed-integer'),
    'u8':  ('<u1', 'unsigned-integer'),
    'u16': ('<u2', 'unsigned-integer'),
    'u32': ('<u4', 'unsigned-integer'),
    'raw': ('<f4', 'float'),
}

def parse_filename(filename):
    # TODO: parse from sigmf.
    match = SAMPLE_FILENAME_RE.match(filename)
    try:
        freq_center = int(match.group(1))
        sample_rate = int(match.group(2))
        sample_type = match.group(3)
    except AttributeError:
        freq_center = None
        sample_rate = None
        sample_type = None
    # FFT is always float not matter the original sample type.
    if is_fft(filename):
        sample_type = 'raw'
    sample_dtype, sample_type = SAMPLE_DTYPES.get(sample_type, (None, None))
    sample_bits = None
    sample_len = None
    if sample_dtype:
        sample_dtype = np.dtype([('i', sample_dtype), ('q', sample_dtype)])
        sample_bits = sample_dtype[0].itemsize * 8
        sample_len = sample_dtype[0].itemsize * 2
    return (freq_center, sample_rate, sample_dtype, sample_len, sample_type, sample_bits)

def get_reader(filename):

    def gzip_reader(x):
        return gzip.open(x, 'rb')

    def zst_reader(x):
        return zstandard.ZstdDecompressor().stream_reader(open(x, 'rb'))

    def default_reader(x):
        return open(x, 'rb')

    if filename.endswith('.gz'):
        return gzip_reader
    if filename.endswith('.zst'):
        return zst_reader

    return default_reader