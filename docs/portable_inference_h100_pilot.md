# Portable inference pilot on H100

This change is based on the latest `cytokineking/Protenix` main at
`b806d2540ab326d71edcc47128343e627248c099`, with the two production
native expected-TM confidence commits carried forward. It was exercised as a
source overlay on the existing ESMFold2 Hyperstack H100 PCIe image. The installed
PyTorch 2.11.0+cu128, native fast LayerNorm, cuEquivariance stack, and
`protenix-v2.pt` checkpoint were unchanged.

Set `PROTENIX_PORTABLE_OPTIMIZATIONS=portable` in the Protenix subprocess to
enable guarded lazy model initialization and prompt release of each completed
prediction. The default is `off`. Lazy initialization is allowed only for
`protenix-v2` with a strict checkpoint load. The inference configuration,
diffusion sampler, seed sequence, template geometry, and confidence calculations
are unchanged. `_PROTENIX_PORTABLE_FEATURES` is an internal ablation setting.

## Representative validation

The fixture is a 125-residue VHH paired with the 115-residue CD274 extracellular
domain, with a predicted complex used as a structural template. It has 240
tokens and 1,903 atoms. The command used 10 recycles, 200 diffusion steps,
one sample, seed 101, no MSA, and full atom confidence. Timings are complete
CLI wall time from `/usr/bin/time -v`; forward time is the runner's existing
log and includes output work. This is a performance and output-compatibility
fixture, not independent binder validation.

| Source/mode | Wall time | Logged forward |
| --- | ---: | ---: |
| Latest fork, off | 80.88 s | 10.03 s |
| Latest fork, off repeat | 79.76 s | 10.06 s |
| Final portable source, lazy initialization + prediction release | 24.16 s | 10.31 s |

The portable run reduced complete single-item wall time by about 70% (3.3x),
almost entirely at startup. Construction plus strict checkpoint load took
60.06 s with ordinary initialization and 1.12 s with lazy initialization.
All 4,174 loaded parameters and buffers had identical SHA-256 hashes after
loading. This is **not** a 3.3x denoising speedup. An A→B→A three-item batch
completed in 38.74 s, so startup savings are amortized across the existing
up-to-10-item CLI grouping.

Both modes produced the same three artifact types and retained native
`token_pair_tm_expected` and `token_pair_tm_normalization_count` fields. Two
off runs differed by 0.048 Å coordinate RMSD and 0.000013 iPTM. The tested
final portable run differed from the first off run by 0.065 Å and 0.000403
iPTM.
The checkpoint-loaded state is exact; final predictions are not bitwise
reproducible on this GPU stack. A larger seed and target panel is still needed
before treating the observed output range as a deployment qualification.

## Features investigated and excluded

An experimental duplicate-template path reused 20 template computations over
10 recycles while preserving the fork's interchain geometry mask. Experimental
diffusion caching recorded 398 cache hits and passed six within-run exact
recomputation checks. On this 240-token case, P1 alone took 24.64 s and logged
10.41 s forward; P1 plus template reuse took 24.26 s / 10.36 s; P1 plus
diffusion caching took 24.11 s / 10.32 s; all combined took 23.91 s / 10.27 s.
These differences are too small to separate confidently from run variation.
The experimental hooks were removed from the deliverable. Selective confidence
allocator retention was also tested in those P1 timings and removed because it
showed no useful gain here.

The source-only patch needs A100 and Blackwell tests before enabling it on
those GPU families or baking new images. The installed image still contains
the earlier Protenix version; an updated image must install the new fork commit
in its separate Protenix environment, preserve the native extension, and run
the pipeline's full validation gate.
