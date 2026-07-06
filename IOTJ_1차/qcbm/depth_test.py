#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""8q underfit 원인 규명: 바렌 플래토 vs 깊이 부족.
   experiment2 파이프라인 재활용, 8큐비트를 L=3/6/9로 비교."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, time
import numpy as np
import experiment2 as E

E.EPOCHS = 300
E.RESTARTS = 16
E.LOG_EVERY = 50

spec8 = {"name": "8q", "nq": 8,
         "features": [("proto", 2), ("syn", 1), ("psh", 1), ("size", 4)]}

benign, attack = E.load_frames()
rng = np.random.default_rng(7)
idx = rng.permutation(len(benign)); cut = int(0.7 * len(benign))
ben_tr = benign.iloc[idx[:cut]].reset_index(drop=True)
edges = E.size_edges_from_benign(ben_tr, 16)
st = E.discretize(ben_tr, spec8, edges)
q = E.empirical_dist(st, 256)
E.log("8q 점유 상태=%d/256" % int((q > 0).sum()))

rows = []
for L in [3, 6, 9]:
    E.LAYERS = L
    E.log("########## 8q  L=%d  ##########" % L)
    co = E.run_config(spec8, q, edges)
    rows.append((L, co["nparams"], co["best"]["final_mmd2"], co["best"]["tv"],
                 co["conv_ep"], co["wall"], co["gnorm0"], co["gnormf"]))

E.log("=" * 64)
E.log("8q 깊이별 비교 (바렌플래토면 깊을수록 grad0가 급감해야 함)")
E.log("  %-3s %-7s %-9s %-7s %-7s %-8s %-9s" %
      ("L", "params", "bestMMD2", "TV", "conv", "wall", "grad0->f"))
for L, p, m, tv, c, w, g0, gf in rows:
    E.log("  %-3d %-7d %-9.2e %-7.3f %-7d %-8.1f %.1e->%.1e" %
          (L, p, m, tv, c, w, g0, gf))
with open("results2/depth_test.json", "w") as f:
    json.dump([dict(L=L, params=p, mmd2=m, tv=tv, conv=c, wall=w, g0=g0, gf=gf)
               for L, p, m, tv, c, w, g0, gf in rows], f, indent=2)
E.log("DONE")
