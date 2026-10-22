# Design decisions

## 1. Exactly-once into the lake without transactions

Kafka transactions only give exactly-once between Kafka topics, while this sink is a set of Parquet files. Here the **lake itself holds the commit log**, as one checkpoint per partition:

```
checkpoint(p) = { next_offset, drained_at, dedup state, window state }
```

An epoch covering offsets `[a, b)` commits in two steps:

1. Each output file is written under a **deterministic** name, `<topic>-p<P>-<a>-<b-1>.parquet`, using tmp + fsync + rename.
2. The checkpoint is atomically replaced, again with tmp + fsync + rename. **This is the commit point.**

Visibility is a pure function of names and checkpoints: a file is committed iff `end < next_offset`, and readers (`lake/catalog.py`) list only committed files. On restart the processor deletes files with `start >= next_offset` and re-consumes from `next_offset`. That covers three crash points:

- **Crash before step 2.** The epoch's files are invisible and get deleted. Replaying regenerates them byte-for-byte under the same names: the processor is deterministic (validity depends on the record's broker timestamp, not the wall clock, and windows close on event time), and the operator state is restored from the same checkpoint.
- **Crash after step 2.** Nothing is lost. Offsets in the consumer group are informational only; positions always come from the checkpoint.
- **Torn file writes.** Writes go through tmp + rename, so a partial file is never visible.

**Tested.** `tests/test_processor.py` crashes at several points (after 1, 4 and 9 batches, between file write and checkpoint, and during a drain) and checks that the lake equals the crash-free run. `tests/test_kafka.py` does the same against a real broker:

- it `kill -9`s a processor mid-stream and restarts it;
- it runs two consumers through a rebalance;
- in both cases it checks that the telemetry multiset, quarantine, DLQ and windows equal a reference built from the same partition logs.

### Drain epochs

A bounded replay must eventually emit the windows that are still open, but closing them consumes no offsets, so it cannot get an ordinary `[a, b)` name. It is a **drain epoch** instead:

- its files are named `<topic>-p<P>-<next_offset>-drain.parquet`;
- the checkpoint records `drained_at = next_offset`;
- a drain file is committed iff `at <= drained_at`.

Two drains cannot happen at the same offset, because the second would find no open windows, so the names are unique. A crash during a drain behaves like any other crash: its files stay invisible and are removed on restart. Both cases are tested.

## 2. Dedup with bounded memory

Remembering every `(vehicle, trip, seq)` grows without bound. Each stream instead keeps:

- `contig`: all seqs up to here have been seen, so in-order traffic costs O(1);
- a sparse set `above` of seqs seen beyond `contig`.

If a gap stays open for more than `dedup_window` seqs, `contig` jumps past it and the skipped range is marked `expired`. A late arrival in an expired range is *ambiguous*: it could be the lost chunk or a duplicate of one already seen. It is dead-lettered with reason `dedup_window_expired`, so it is neither dropped silently nor allowed to duplicate rows. Finished streams are evicted by event-time TTL.

## 3. Per-stream watermarks

One global watermark would let a vehicle uploading over a slow link hold back every other vehicle, or close another vehicle's windows too early. Each `(vehicle, trip)` stream therefore has its own watermark: `max event time − allowed lateness`.

Streams that stop sending are closed when the **partition's event-time frontier** passes them by `idle_timeout`, never by wall clock. This is required for deterministic replay. The consequence is that window output depends on which streams share a partition, so the Kafka tests build their reference from the actual partition logs, not from an idealised split.

Late rows still go to `telemetry`, the source of truth. They are excluded from `windows_10s` and counted. The batch metrics recompute everything from `telemetry`, and `tests/test_metrics.py` checks that streaming and batch distance agree to within 1%.

## 4. Two failure granularities

- **A broken chunk goes to the DLQ** (corrupt payload, unknown schema version, missing identity, mismatched column lengths). It is stored exactly-once in the lake's `dlq` table, and also sent at-least-once to `telemetry.dlq` with a `reason` header for alerting.
- **A broken sample goes to quarantine** (out of physical bounds, NaN in a required signal, time going backwards, too far in the future). Only that row is set aside, together with a bitmask of every rule it failed. A single spike must not cost the other 19 samples of its chunk.

Missing *optional* signals (no GNSS fix, no lead vehicle) are not errors. They travel as NaN on the wire, so every column has the same length, and are stored as SQL `NULL` so that `count(lat)` and `avg(lead_dist)` mean what they say.

## 5. Detectors validated against ground truth

Detectors are vectorised run-length logic over each trip:

1. a predicate per sample;
2. a minimum duration;
3. gap merging.

Severity maps `threshold → extreme` linearly onto `[0.2, 1.0]`, so severities are comparable across scenario types. The thresholds are listed in `scenarios/detectors.py`.

The synthetic fleet exists to make detector quality **measurable**. Every injected scenario has a ground-truth label, and `evaluate()` reports precision and recall per type. Two intensities are tested:

- **`clear`.** Every injected event is unambiguous. The test requires perfect precision and recall even with realistic transport faults.
- **`borderline`.** Event magnitudes straddle the thresholds. Precision must stay at 100%, and recall must land strictly between 0.3 and 0.9. Recall of exactly 1.0 would mean the thresholds are not doing anything.

Getting these tests to pass found real modelling problems:

- **False low-TTC alarms.** An earlier lead model differentiated a noisy gap, which produced false alarms. It was replaced by a lagged-gap car-following model with independent radar range and range-rate noise.
- **Tailgating after cut-ins.** Close following right after a cut-in was reported as tailgating. That following distance is imposed on the driver, not chosen, so it is now masked for 10 s after a close lane entry.

## 6. Ranking: severity × rarity

`priority = ½·severity + ½·rarity`, where

`rarity = ½·type surprisal + ½·within-type novelty`.

- **Type surprisal** is `−log p(type)`, normalised.
- **Novelty** is the percentile rank of the distance to the k-th nearest neighbour of the same type, in robust-standardised feature space.

So a 25 m/s hard brake that looks like 500 others ranks below one at 3 m/s with extreme jerk.

## 7. Selection under a storage budget

Uploading every clip is impossible, and taking the top N by priority spends the budget on near-duplicates. The selection maximises a **type-balanced facility-location** objective:

```
f(S) = Σ_i w_i · max_{j∈S} sim(i, j),   sim = exp(−‖x_i − x_j‖² / 2σ²) within a type, 0 across
w_i  = v_t · p_i / Σ_{j∈t} p_j,          v_t = mean priority of type t
```

A clip earns value by *representing* many valuable, similar scenarios. The weights stop the most common type from dominating just because it is common, and they make `f(S)/f(all)` read as "the priority-weighted share of each type's variety that was captured".

Clip cost is `(duration + 2·padding) × bytes/s`, which makes this a knapsack constraint. The selector returns the better of:

1. **cost-benefit lazy greedy (CELF)**, which picks the largest marginal gain per byte; and
2. **the best single clip that fits.**

Taking the better of the two is what gives the **(1 − 1/e)/2 guarantee**. Greedy alone has no constant-factor bound under a knapsack constraint; `test_selection.py` includes the classic counterexample, where a small clip with a slightly better ratio crowds out a clip worth 5× more.

The tests check four things:

- lazy and plain greedy pick identical sets;
- on brute-forced small instances, every result meets the bound, and the typical ratio is above 0.95;
- every method respects the budget;
- diverse selection beats the top-priority, top-severity, stratified and random baselines on coverage.

## 8. Metrics

Distance integrates speed only across steps of 0.2 s or less, so a dropout never adds distance that was not observed. Fleet ratios are aggregated with distance or time weighting, never as a mean of per-trip ratios, which would over-weight short trips. Full definitions are in [metrics.md](metrics.md).

## Known limitations

- **The fencing race.** A zombie processor is caught by the `next_offset` compare-and-set at commit time. A zombie that passes that check just before the new owner commits could still rename files afterwards. Two things would close this fully: conditional writes on an object store, and listing committed epochs in the checkpoint instead of deriving visibility from offsets. The window is small: the old owner must outlive its session timeout *and* be mid-commit. Within the scope of this project the gap is documented rather than fixed.
- **Selection memory.** Similarity matrices are dense per type, O(m²) for m clips of one type. That is fine up to tens of thousands of candidates per run; beyond that, a sparse kNN graph or a stochastic greedy would be needed.
- **One node.** The demo stack is a single broker without replication. Topic settings (`acks=all`, idempotence, LogAppendTime) are production-shaped, but replication factor and `min.insync.replicas` would need to be raised.
- **Synthetic labels.** Detector accuracy is only measured against synthetic ground truth. On comma2k19, the test checks only that clean real driving produces no false safety events.
