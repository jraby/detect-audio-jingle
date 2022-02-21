`detect-audio-jingle.py` is a script that tries to locate a small audio clip (`-c`) in
longer audio files (`-i`, `-d`).
If the clip has been found, a new audio file starting at the position of the
audio clip in the input file will be created in the directory specified by `--output-dir`.
(otherwise it will output the input file itself)

For example, this can be used to skip over ads in some audio files / podcast episodes when there's a
distinct "jingle" that signals the beginning of the episode.

The algorithm used is not very sophisticated and probably not very robust, but it worked well for my
purposes.


It uses [fftconvolve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html) to locate the clip,
and it was inspired by [this stackoverflow answer](https://stackoverflow.com/a/52682780).

# Installation

```
virtualenv env
. env/bin/activate
pip install -r requirements.txt

```

# Running

```
./detect-audio-jingle.py -c ~/tmp/revolutions-jingle.mp3 \
                         -i ~/tmp/The_Streets_of_Paris_Master.mp3  \
                         --output-dir ~/tmp/output

/home/jean/tmp/The_Streets_of_Paris_Master.mp3                  : KEEP 44100hz event_detection_threshold=98.56 fp_max_threshold=92.11 zmax=425.35 zstd=5.41 zmean=5.50 z_event_start=3134731 seek:71.08s
ffmpeg version n5.0 Copyright (c) 2000-2022 the FFmpeg developers
  built with gcc 11.2.0 (GCC)
...
```
