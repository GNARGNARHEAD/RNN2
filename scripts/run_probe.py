#!/usr/bin/env python3
"""Run a native probe with bounded time and record host/device telemetry.

Python only supervises the process; all model computation is in the C++ binary.
Whole-device memory includes the desktop and is not the probe's allocation.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def read_int(path):
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def meminfo():
    return {line.split(':')[0]: int(line.split()[1]) * 1024
            for line in Path('/proc/meminfo').read_text().splitlines()
            if len(line.split()) == 3 and line.split()[2] == 'kB'}


def sample(pid, start, gpu):
    memory = meminfo()
    try:
        status = Path(f'/proc/{pid}/status').read_text().splitlines()
        rss = next(int(s.split()[1]) * 1024 for s in status if s.startswith('VmRSS:'))
    except (OSError, StopIteration):
        rss = 0
    sensors = {}
    for hwmon in Path('/sys/class/hwmon').glob('hwmon*'):
        try:
            name = (hwmon / 'name').read_text().strip()
        except OSError:
            continue
        if name in ('amdgpu', 'k10temp'):
            sensors[name + '_millidegrees_c'] = read_int(hwmon / 'temp1_input')
    return dict(elapsed_s=round(time.monotonic() - start, 3), rss_bytes=rss,
                mem_available_bytes=memory['MemAvailable'],
                swap_used_bytes=memory['SwapTotal'] - memory['SwapFree'],
                gpu_vram_used_bytes=read_int(gpu / 'mem_info_vram_used'),
                gpu_gtt_used_bytes=read_int(gpu / 'mem_info_gtt_used'),
                gpu_busy_percent=read_int(gpu / 'gpu_busy_percent'),
                **sensors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=240)
    parser.add_argument('--min-available-mib', type=int, default=1536)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.timeout <= 0 or args.min_available_mib < 0:
        parser.error('provide a command, positive timeout, and nonnegative memory reserve')
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in ('stdout.log', 'telemetry.csv', 'result.json')):
        parser.error('output directory already contains run artifacts; choose a fresh directory')
    gpu_candidates = sorted(Path('/sys/class/drm').glob('card[0-9]*/device/mem_info_vram_used'))
    gpu = gpu_candidates[0].parent if gpu_candidates else Path('/nonexistent')
    start = time.monotonic()
    baseline = sample(os.getpid(), start, gpu)
    if baseline['mem_available_bytes'] < args.min_available_mib * 1024**2:
        parser.error('not enough available memory to start with requested reserve')
    result = dict(command=command, baseline=baseline,
                  min_available_mib=args.min_available_mib, timeout_s=args.timeout)
    with (args.output / 'stdout.log').open('w') as log, (args.output / 'telemetry.csv').open('w') as telemetry:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        writer = csv.DictWriter(telemetry, fieldnames=list(baseline))
        writer.writeheader()
        peak_rss = 0
        min_available = baseline['mem_available_bytes']
        reason = None
        try:
            while process.poll() is None:
                row = sample(process.pid, start, gpu)
                writer.writerow(row)
                telemetry.flush()
                peak_rss = max(peak_rss, row['rss_bytes'])
                min_available = min(min_available, row['mem_available_bytes'])
                if row['elapsed_s'] >= args.timeout:
                    reason = 'timeout'
                elif row['mem_available_bytes'] < args.min_available_mib * 1024**2:
                    reason = 'memory_reserve'
                if reason:
                    break
                time.sleep(0.5)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            result.update(returncode=process.returncode, stopped_by_guard=reason,
                          elapsed_s=time.monotonic() - start,
                          peak_sampled_rss_bytes=peak_rss,
                          min_available_bytes=min_available)
            (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(0 if result['returncode'] == 0 and not reason else 1)


if __name__ == '__main__':
    main()
