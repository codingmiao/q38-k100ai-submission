#!/usr/bin/env python3
"""从 benchmark_official.py 的 round-*/requests.jsonl 计算全局均值 TTFT/TPOT。

官方 harness 的 aggregate 只报 p50/p95（跨轮），不报均值。宽松 SLA 口径
（组委会"取均值性能"）判分用均值 TTFT/TPOT，故从原始 per-request 记录重算。

用法: python3 sla_mean_postproc.py <output_dir>
读 <output_dir>/round-*/requests.jsonl（measure 轮），输出:
  ttft_mean_s / ttft_p95_s / tpot_mean_ms / tpot_p95_ms / n
"""
import glob
import json
import os
import sys


def pct(v, p):
    v = sorted(v)
    if not v:
        return None
    if len(v) == 1:
        return v[0]
    pos = (len(v) - 1) * p / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def main():
    d = sys.argv[1]
    ttfts, tpots = [], []
    files = sorted(glob.glob(os.path.join(d, "round-*", "requests.jsonl")))
    for f in files:
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("ttft_s") is not None:
                ttfts.append(r["ttft_s"])
            if r.get("tpot_ms") is not None:
                tpots.append(r["tpot_ms"])
    out = {
        "rounds": len(files),
        "n_ttft": len(ttfts),
        "n_tpot": len(tpots),
        "ttft_mean_s": (sum(ttfts) / len(ttfts)) if ttfts else None,
        "ttft_p95_s": pct(ttfts, 95),
        "tpot_mean_ms": (sum(tpots) / len(tpots)) if tpots else None,
        "tpot_p95_ms": pct(tpots, 95),
    }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
