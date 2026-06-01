#!/usr/bin/env python3
"""
Analyse rapide des fichiers de timing pour identifier le bottleneck
"""
import re
import numpy as np

def parse_timing_line(line):
    """Parse une ligne de timing et extrait les valeurs"""
    pattern = r'TOTAL=(\d+\.?\d*)ms.*x1_load=(\d+\.?\d*)ms.*x10=(\d+\.?\d*)ms.*x3=(\d+\.?\d*)ms.*valid=(\d+\.?\d*)ms.*postpro=(\d+\.?\d*)ms'
    match = re.search(pattern, line)
    if match:
        return {
            'total': float(match.group(1)),
            'x1_load': float(match.group(2)),
            'x10': float(match.group(3)),
            'x3': float(match.group(4)),
            'valid': float(match.group(5)),
            'postpro': float(match.group(6))
        }
    return None

def analyze_timings(log_file):
    """Analyse le fichier de timing multi-résolution"""
    timings = []

    with open(log_file, 'r') as f:
        for line in f:
            if 'x1_load=' in line:
                parsed = parse_timing_line(line)
                if parsed:
                    timings.append(parsed)

    if not timings:
        print("Aucun timing trouvé !")
        return

    # Convertir en numpy arrays pour stats
    total = np.array([t['total'] for t in timings])
    x1_load = np.array([t['x1_load'] for t in timings])
    x10 = np.array([t['x10'] for t in timings])
    x3 = np.array([t['x3'] for t in timings])
    valid = np.array([t['valid'] for t in timings])
    postpro = np.array([t['postpro'] for t in timings])

    print("="*80)
    print(f"ANALYSE DE {len(timings)} SAMPLES MULTI-RÉSOLUTION")
    print("="*80)
    print()

    print("TEMPS TOTAL PAR SAMPLE (ms)")
    print(f"  Min      : {total.min():.1f} ms = {total.min()/1000:.2f} s")
    print(f"  Max      : {total.max():.1f} ms = {total.max()/1000:.2f} s")
    print(f"  Moyenne  : {total.mean():.1f} ms = {total.mean()/1000:.2f} s")
    print(f"  Médiane  : {np.median(total):.1f} ms = {np.median(total)/1000:.2f} s")
    print(f"  Écart-type: {total.std():.1f} ms")
    print(f"  P95      : {np.percentile(total, 95):.1f} ms = {np.percentile(total, 95)/1000:.2f} s")
    print(f"  RATIO MAX/MIN: {total.max() / total.min():.1f}x")
    print()

    print("DÉCOMPOSITION MOYENNE (en % du temps total)")
    print(f"  x1_load  : {x1_load.mean():.1f} ms ({100*x1_load.mean()/total.mean():.1f}%)")
    print(f"  x10      : {x10.mean():.1f} ms ({100*x10.mean()/total.mean():.1f}%)")
    print(f"  x3       : {x3.mean():.1f} ms ({100*x3.mean()/total.mean():.1f}%)")
    print(f"  valid    : {valid.mean():.1f} ms ({100*valid.mean()/total.mean():.1f}%)")
    print(f"  postpro  : {postpro.mean():.1f} ms ({100*postpro.mean()/total.mean():.1f}%)")
    print()

    print("CHARGEMENT X1 (résolution fine)")
    print(f"  Min      : {x1_load.min():.1f} ms = {x1_load.min()/1000:.2f} s")
    print(f"  Max      : {x1_load.max():.1f} ms = {x1_load.max()/1000:.2f} s")
    print(f"  Moyenne  : {x1_load.mean():.1f} ms = {x1_load.mean()/1000:.2f} s")
    print(f"  Médiane  : {np.median(x1_load):.1f} ms = {np.median(x1_load)/1000:.2f} s")
    print(f"  RATIO MAX/MIN: {x1_load.max() / x1_load.min():.1f}x")
    print()

    print("CHARGEMENT X10 (résolution grossière)")
    print(f"  Min      : {x10.min():.1f} ms = {x10.min()/1000:.2f} s")
    print(f"  Max      : {x10.max():.1f} ms = {x10.max()/1000:.2f} s")
    print(f"  Moyenne  : {x10.mean():.1f} ms = {x10.mean()/1000:.2f} s")
    print(f"  RATIO MAX/MIN: {x10.max() / x10.min():.1f}x")
    print()

    print("CHARGEMENT X3 (résolution intermédiaire)")
    print(f"  Min      : {x3.min():.1f} ms = {x3.min()/1000:.2f} s")
    print(f"  Max      : {x3.max():.1f} ms = {x3.max()/1000:.2f} s")
    print(f"  Moyenne  : {x3.mean():.1f} ms = {x3.mean()/1000:.2f} s")
    print(f"  RATIO MAX/MIN: {x3.max() / x3.min():.1f}x")
    print()

    # Distribution par buckets
    print("DISTRIBUTION DES TEMPS TOTAUX:")
    buckets = [0, 1000, 2000, 3000, 4000, 5000, 100000, 200000]
    labels = ["< 1s", "1-2s", "2-3s", "3-4s", "4-5s", "5-10s", "10-20s", "> 20s"]
    hist, _ = np.histogram(total, bins=buckets + [np.inf])

    for label, count in zip(labels, hist):
        pct = 100 * count / len(total)
        bar = "█" * int(pct / 2)
        print(f"  {label:10s}: {count:4d} samples ({pct:5.1f}%) {bar}")

if __name__ == '__main__':
    analyze_timings('timings_multires.log')
