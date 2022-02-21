#!/bin/env python3
import argparse
import glob
import librosa
import ffmpeg
import numpy as np
import os.path
import scipy.signal
import shutil
import warnings

from natsort import natsorted


max_longclip_duration =  4 * 60


def loadAudio(filename, sr=None):
    y, sr = librosa.load(filename, duration=max_longclip_duration, mono=True, sr=sr)
    return y, sr


def save_trimmed(in_file, out_file, seek):
    if not seek:
        shutil.copyfile(in_file, out_file)
        return

    ffmpeg.input(in_file, ss=seek).output(out_file, acodec="copy").overwrite_output().run()


def main():
    warnings.filterwarnings('ignore', category=UserWarning, append=True)
    parser = argparse.ArgumentParser()
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-d", "--input-dir", help="process all mp3s from this directory", action="store")
    input_group.add_argument("-i", "--input-file", help="mp3 file to process", action="store")
    parser.add_argument("-c", "--clip", help="mp3 clip to try and locate in input file(s)",
            required=True)
    parser.add_argument("--output-dir", help="Directory in which to save trimmed mp3s")
    parser.add_argument("-n", "--dry-run", help="Dry-run", action="store_true")
    parser.add_argument("--plots-dir", help="Directory in which to save a plot for each detection")
    parser.add_argument("--fp-threshold-factor", default=16, type=int,
            help="false positive threshold factor: max peak must be > [factor] * stddev + mean to trigger detection")
    parser.add_argument("--percentile", help="First sample >= the percentile is considered 'peak'",
            default=99.99, type=float)  # determined by precise eye balling
    args = parser.parse_args()

    if not args.output_dir and not args.dry_run:
        raise Exception("Needs --output-dir or --dry-run")

    clip, clip_sr = loadAudio(args.clip)
    # Cache of sample_rate -> reversed clip to be used by fftconvolve
    clip_sr_cache = {clip_sr: clip[::-1]}

    if args.input_file:
        input_files = [args.input_file]
    else:
        input_files = natsorted(glob.glob(f"{args.input_dir}/*.mp3"))

    for f in input_files:
        base_f = os.path.basename(f)

        if args.output_dir and not args.dry_run:
            out_file = args.output_dir + '/' + base_f
            if os.path.exists(out_file):
                print(f"{f[:64]:64}: SKIPPED - output file already exists")
                continue

        f_samples, f_sr = loadAudio(f)
        if f_sr not in clip_sr_cache:
            # Resample clip to match current file and cache it (reversed) for future use
            clip_sr_cache[f_sr] = librosa.core.resample(clip, clip_sr, f_sr)[::-1]
        to_find = clip_sr_cache[f_sr]

        # Find clip in input file using fftconvolve. Then approximate the start position of the clip.
        # The 'peak' of the result is supposed to be the middle of the match, but it seems to be a
        # little late according to precise by ear measurement.
        # So instead of using the max value, this is using the first value that is in some high
        # percentile (99.99).
        z = scipy.signal.fftconvolve(f_samples, to_find, mode="same")
        z = np.abs(z)

        event_detection_threshold = np.percentile(z, args.percentile)
        z_event_pos = np.argmax(z >= event_detection_threshold)
        z_event_start = z_event_pos - len(to_find) // 2  # (peak assumed to be in middle of clip)
        # But using percentiles leads to early detection, needs to floor to 0
        z_event_start = z_event_start if z_event_start > 0 else 0

        seek = z_event_start / f_sr

        zmax = np.max(z)
        zstd = np.std(z)
        zmean = np.mean(z)
        fp_max_threshold = zstd * args.fp_threshold_factor + zmean

        keep = zmax > fp_max_threshold and seek > 10

        if not keep:
            seek = 0

        print((f"{f[:64]:64}: {'KEEP' if keep else 'NOPE'} {f_sr}hz "
            f"{event_detection_threshold=:0.2f} {fp_max_threshold=:0.2f} "
            f"{zmax=:0.2f} {zstd=:0.2f} {zmean=:0.2f} {z_event_start=} "
            f"seek:{seek:0.2f}s"))

        if args.plots_dir:
            import matplotlib.pyplot as plt
            os.makedirs(args.plots_dir, exist_ok=True)
            x_min = 0
            x_max = z_event_start + 5*f_sr
            if x_max > len(z):
                x_max = len(z)
            plt.plot(z[x_min:x_max])
            plt.xlabel("Sample number")
            plt.title(base_f)
            plt.savefig(f"{args.plots_dir}/{os.path.basename(f)}.png")
            plt.clf()

        if args.output_dir and not args.dry_run:
            os.makedirs(args.output_dir, exist_ok=True)
            out_file = args.output_dir + '/' + base_f
            save_trimmed(in_file=f, out_file=out_file, seek=seek)

if __name__ == '__main__':
    main()
