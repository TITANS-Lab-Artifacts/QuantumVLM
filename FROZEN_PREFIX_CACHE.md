# Frozen-prefix caching

## Scope

Only quantum insertion `quantum_only` at q1, q2, q3 gains a new execution path. The base model must be entirely frozen and in evaluation mode. q4's original final-feature cache is retained. All circuits, locations, training hyperparameters, mixed-precision settings, optimizer, scheduler, and checkpoint tensors are unchanged. Replacement training and joint training do not use this cache. No multi-GPU scheduling was added.

## Architecture

| Position | Stored tensor | Recomputed with gradients |
|---|---|---|
| q1 | Input to the visual projector, before the adapter | Adapter, projector, image packing, decoder, scoring |
| q2 | Visual projector output, before the adapter | Adapter, image packing, decoder, scoring |
| q3 | Complete sequence after the selected frozen decoder block, before the adapter | Adapter at the final prompt position, remaining decoder blocks, normalization, scoring |
| q4 | Existing final prompt feature cache | Existing adapter and scoring path |

q3 stores the full sequence, not just the final token. The native decoder rebuilds masks and rotary positions, while the frozen prefix blocks temporarily return their supplied hidden states. The original insertion hook runs once at the original location. The temporary overrides are restored in a finally block. These model instances must not be scored from multiple threads simultaneously. The sequential runner satisfies this restriction.

q1/q2 use the native OneVision packing, placeholder-mask construction, and Qwen decoder. Each sample has one image as in the existing VSR pipeline; multi-image/video inputs are not supported. The implementation targets the pinned Transformers 4.57.6 model structure.

## Disk and memory

Prefix tensors preserve their original dtype. One tensor file per example has a SHA-256 sidecar. Files are written through a temporary path and reused only when their checksums match. Cache identity includes model settings, insertion position, decoder index, environment, source hashes, manifest, sample identity, input-image fingerprint, and prompt fingerprint. Learned adapter weights are intentionally excluded because they are downstream of the cache boundary. Only paths, token IDs, masks and metadata stay in RAM; one prefix is loaded for each training example. Disk requirements depend on actual sequence lengths and can exceed RAM substantially. No performance or disk-capacity claim is inferred from qubit count.

## Correctness and resume

Cache preparation compares original and cached scores, cross-entropy loss and all trainable gradients on the configured verification examples of both train and dev. Failure aborts. This is a sampled runtime safeguard, not proof over every possible input. The best model is still evaluated and reload-verified through the original uncached path. Attack and diagnostic code never receives cached features; image gradients remain intact.

`migrate_prefix_resume.py` changes only the registry source signature for exactly the supported previous release, storing the original registry and a migration record. Configuration/environment/data checks remain strict. Completed results are not rewritten. Incomplete quantum-only runs retain model/optimizer/scheduler/RNG/shuffle state. A source migration does not itself certify numerical equivalence: runtime checks must still pass. Full-size CUDA bitwise equivalence and speed are not guaranteed by CPU tests.

Timing is separated into cache preparation and training; each new epoch reports its execution path. A run migrated halfway through has mixed cumulative timing. Retain per-session resource files and the migration record when publishing results.

## Validation

The cache was checked using small real LLaVA-OneVision/Qwen2 models and the four PyTorch circuit implementations, not a simulated training-output generator. The tests are not part of this repository. See VALIDATION.md for the completed checks and their hardware limitations. Full-size GPU runs remain the final validation environment, and every run repeats the runtime equivalence checks described above.
