#!/usr/bin/env python3
"""
split_m4b.py - split an .m4b audiobook into numbered files for a watch/phone.

Reads the embedded chapter list with ffprobe, optionally merges tiny chapters,
optionally sub-splits long chapters (optionally snapping cuts to silence),
then cuts each segment with ffmpeg (lossless stream copy by default).

Requires: Python 3.8+, ffmpeg and ffprobe on PATH. No third-party packages.

Examples:
  split_m4b.py book.m4b --dry-run
  split_m4b.py book.m4b -o out/ --max-minutes 10 --snap-silence
  split_m4b.py book.m4b --max-minutes 10 --format mp3 --cover
"""

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass
class Segment:
    start: float          # seconds
    end: float            # seconds
    title: str

    @property
    def duration(self) -> float:
        return self.end - self.start


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def hms(seconds: float) -> str:
    """Format seconds as h:mm:ss.mmm (also accepted by ffmpeg)."""
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h}:{m:02d}:{s:02d}.{ms:03d}"


def sanitize(name: str, max_len: int = 80) -> str:
    """Make a string safe for FAT/NTFS/ext4 filenames."""
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', " ", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    return name[:max_len].rstrip(" .") or "untitled"


def die(msg: str, code: int = 1):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def check_tools():
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            die(f"{tool} not found on PATH. Install ffmpeg first.")


# --------------------------------------------------------------------------- #
# Probing
# --------------------------------------------------------------------------- #

def probe(path: Path) -> dict:
    cmd = ["ffprobe", "-v", "error", "-print_format", "json",
           "-show_chapters", "-show_format", str(path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        die(f"ffprobe failed:\n{res.stderr.strip()}")
    return json.loads(res.stdout)


def chapters_from_probe(info: dict, duration: float):
    segs = []
    for i, ch in enumerate(info.get("chapters", [])):
        if "start_time" in ch and "end_time" in ch:
            start, end = float(ch["start_time"]), float(ch["end_time"])
        else:  # fall back to raw ticks + time_base
            tb = Fraction(ch.get("time_base", "1/1000"))
            start, end = float(int(ch["start"]) * tb), float(int(ch["end"]) * tb)
        end = min(end, duration)
        title = (ch.get("tags") or {}).get("title") or f"Chapter {i + 1}"
        title = title.replace(' ','_')
        if end > start:
            segs.append(Segment(start, end, title.strip()))
    return segs


# --------------------------------------------------------------------------- #
# Segment-list transforms (the script's single source of truth)
# --------------------------------------------------------------------------- #

def merge_short(segs, min_seconds: float):
    """Merge segments shorter than min_seconds into a neighbour."""
    if min_seconds <= 0 or len(segs) < 2:
        return segs
    out = []
    for seg in segs:
        if out and seg.duration < min_seconds:
            out[-1].end = seg.end                 # absorb into previous
        else:
            out.append(Segment(seg.start, seg.end, seg.title))
    # A short *first* segment: absorb into the next one.
    if len(out) > 1 and out[0].duration < min_seconds:
        out[1].start = out[0].start
        out.pop(0)
    return out


def detect_silences(path: Path, noise_db: float, min_dur: float):
    """Return midpoints (seconds) of all silences in the first audio stream."""
    print(f"Running silence detection ({noise_db} dB, >= {min_dur}s). "
          f"This decodes the whole file and can take a few minutes...")
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-map", "0:a:0",
           "-af", f"silencedetect=noise={noise_db}dB:d={min_dur}", "-f", "null", "-"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", res.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", res.stderr)]
    mids = [(max(s, 0.0) + e) / 2 for s, e in zip(starts, ends)]
    print(f"  found {len(mids)} silences")
    return mids


def subsplit(segs, max_seconds: float, silences=None, window: float = 20.0):
    """
    Divide any segment longer than max_seconds into roughly equal parts.
    If `silences` is given, move each internal cut to the nearest silence
    within +/- window seconds.
    """
    if not max_seconds:
        return segs
    out = []
    for seg in segs:
        if seg.duration <= max_seconds:
            out.append(seg)
            continue
        n = math.ceil(seg.duration / max_seconds)
        step = seg.duration / n
        cuts = []
        for k in range(1, n):
            t = seg.start + k * step
            if silences:
                near = [s for s in silences
                        if abs(s - t) <= window and seg.start + 1 < s < seg.end - 1]
                if near:
                    t = min(near, key=lambda s: abs(s - t))
            cuts.append(t)
        cuts = sorted(set(cuts))
        bounds = [seg.start] + cuts + [seg.end]
        parts = len(bounds) - 1
        for p in range(parts):
            out.append(Segment(bounds[p], bounds[p + 1],
                               f"{seg.title} (part {p + 1} of {parts})"))
    return out


# --------------------------------------------------------------------------- #
# Cutting
# --------------------------------------------------------------------------- #

def extract_cover(src: Path, dest: Path, size: int) -> bool:
    """Pull the embedded cover once, downscaled, to reuse for every file."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
           "-map", "0:v:0", "-frames:v", "1",
           "-vf", f"scale={size}:{size}:force_original_aspect_ratio=decrease",
           str(dest)]
    return subprocess.run(cmd, capture_output=True).returncode == 0 and dest.exists()


def build_cut_cmd(src, seg, tmp_out, fmt, tags, cover, mp3_kbps):
    # -ss before -i is fast, and accurate for audio-only stream copy.
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-ss", f"{seg.start:.3f}", "-t", f"{seg.duration:.3f}", "-i", str(src)]
    if cover:
        cmd += ["-i", str(cover)]
    cmd += ["-map", "0:a:0"]
    if cover:
        cmd += ["-map", "1:v", "-c:v", "copy", "-disposition:v:0", "attached_pic"]
    cmd += ["-map_metadata", "-1", "-map_chapters", "-1",
            "-avoid_negative_ts", "make_zero"]
    if fmt == "mp3":
        cmd += ["-c:a", "libmp3lame", "-b:a", f"{mp3_kbps}k", "-id3v2_version", "3"]
        muxer = "mp3"
    else:
        cmd += ["-c:a", "copy"]
        muxer = "ipod"
    for key, val in tags.items():
        if val:
            cmd += ["-metadata", f"{key}={val}"]
    cmd += ["-f", muxer, str(tmp_out)]
    return cmd


def file_duration(path: Path) -> float:
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True)
    try:
        return float(res.stdout.strip())
    except ValueError:
        return 0.0


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Split an m4b audiobook by chapter.")
    p.add_argument("input", type=Path)
    p.add_argument("-o", "--output-dir", type=Path,
                   help="default: <input name>_split next to the input")
    p.add_argument("--max-minutes", type=float, default=0,
                   help="sub-split any chapter longer than this (0 = off)")
    p.add_argument("--min-seconds", type=float, default=60,
                   help="merge chapters shorter than this into a neighbour "
                        "(default 60; 0 = off)")
    p.add_argument("--snap-silence", action="store_true",
                   help="snap sub-split cuts to nearby silence")
    p.add_argument("--silence-db", type=float, default=-35,
                   help="silence threshold in dB (default -35)")
    p.add_argument("--silence-min", type=float, default=0.5,
                   help="minimum silence length in seconds (default 0.5)")
    p.add_argument("--snap-window", type=float, default=20,
                   help="max seconds a cut may move to reach silence (default 20)")
    p.add_argument("--format", choices=["m4a", "mp3"], default="m4a",
                   help="m4a = lossless copy (default); mp3 = re-encode")
    p.add_argument("--mp3-bitrate", type=int, default=96,
                   help="kbps for --format mp3 (default 96)")
    p.add_argument("--cover", action="store_true",
                   help="embed a downscaled cover image in each file")
    p.add_argument("--cover-size", type=int, default=200,
                   help="cover max dimension in pixels (default 200)")
    p.add_argument("--album", help="override album tag")
    p.add_argument("--artist", help="override artist tag")
    p.add_argument("--year", help="date tag (source 'date' is NOT copied)")
    p.add_argument("--dry-run", action="store_true", help="print plan and exit")
    return p.parse_args()


def main():
    args = parse_args()
    check_tools()
    src = args.input
    if not src.is_file():
        die(f"input not found: {src}")

    info = probe(src)
    fmt_info = info.get("format", {})
    duration = float(fmt_info.get("duration", 0))
    book_tags = {k.lower(): v for k, v in (fmt_info.get("tags") or {}).items()}
    if duration <= 0:
        die("could not read file duration")

    # ---- 1. chapters -> segments -------------------------------------------
    segs = chapters_from_probe(info, duration)
    if not segs:
        if args.max_minutes:
            print("No chapters found; splitting the whole file by --max-minutes.")
            segs = [Segment(0.0, duration, book_tags.get("title", src.stem))]
        else:
            die("no chapters found. Re-run with --max-minutes N to split by length.")

    # ---- 2. transforms ------------------------------------------------------
    segs = merge_short(segs, args.min_seconds)
    silences = None
    if args.max_minutes and args.snap_silence and not args.dry_run:
        silences = detect_silences(src, args.silence_db, args.silence_min)
    segs = subsplit(segs, args.max_minutes * 60, silences, args.snap_window)

    total = len(segs)
    width = max(2, len(str(total)))

    album = args.album or book_tags.get("album") or book_tags.get("title") or src.stem
    artist = args.artist or book_tags.get("artist") or book_tags.get("album_artist")
    album_artist = book_tags.get("album_artist") or artist
    genre = book_tags.get("genre", "Audiobook")

    outdir = args.output_dir or src.with_name(src.stem + "_split")
    ext = args.format

    # ---- 3. plan ------------------------------------------------------------
    plan = []
    for i, seg in enumerate(segs, 1):
        fname = f"{src.stem}_{i:0{width}d}-{sanitize(seg.title)}.{ext}"
        plan.append((i, seg, outdir / fname))

    print(f"\nInput:   {src.name}  ({hms(duration)})")
    print(f"Output:  {outdir}  ({total} files, .{ext})\n")
    for i, seg, path in plan:
        print(f"  {i:>{width}}  {hms(seg.start)} -> {hms(seg.end)}  "
              f"({seg.duration / 60:5.1f} min)  {path.name}")
    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    # ---- 4. cut -------------------------------------------------------------
    outdir.mkdir(parents=True, exist_ok=True)
    cover = None
    if args.cover:
        cover = outdir / ".cover.jpg"
        if not extract_cover(src, cover, args.cover_size):
            print("warning: no cover art found; continuing without it.")
            cover = None

    failures = []
    print()
    for i, seg, path in plan:
        if path.exists() and path.stat().st_size > 0:
            print(f"[{i}/{total}] skip (exists): {path.name}")
            continue
        print(f"[{i}/{total}] {path.name}")
        tags = {"title": seg.title, "track": f"{i}/{total}", "album": album,
                "artist": artist, "album_artist": album_artist,
                "genre": genre, "date": args.year}
        tmp = path.with_name(path.name + ".part")
        cmd = build_cut_cmd(src, seg, tmp, ext, tags, cover, args.mp3_bitrate)
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 or not tmp.exists():
            failures.append((path.name, res.stderr.strip()))
            tmp.unlink(missing_ok=True)
            print(f"   FAILED: {res.stderr.strip().splitlines()[-1] if res.stderr.strip() else 'unknown error'}")
            continue
        tmp.replace(path)

    if cover:
        cover.unlink(missing_ok=True)

    # ---- 5. verify ----------------------------------------------------------
    produced = [p for _, _, p in plan if p.exists()]
    out_total = sum(file_duration(p) for p in produced)
    expected = sum(s.duration for _, s, _ in plan)
    print(f"\nDone: {len(produced)}/{total} files.")
    print(f"Duration check: output {hms(out_total)} vs expected {hms(expected)} "
          f"(diff {out_total - expected:+.2f}s)")
    if failures:
        print(f"\n{len(failures)} failure(s):")
        for name, err in failures:
            print(f"  - {name}: {err[:200]}")
        sys.exit(2)


if __name__ == "__main__":
    main()
