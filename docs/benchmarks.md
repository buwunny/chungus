# Benchmarks

`bench/run.py` measures what chungus saves on real models and how fast it downloads them, next to Hugging Face and its Xet storage. It needs [uv](https://docs.astral.sh/uv/) and a Rust toolchain; uv puts `huggingface_hub` in a throwaway environment, so nothing is installed globally.

```sh
uv run bench/run.py               # quick set: about 2 GB of downloads
uv run bench/run.py --set full    # adds Qwen2.5/Qwen3, FP8, GGUF Q8 and TinyLlama 1.1B
```

It builds `target/release/chungus`, downloads the models once into `bench/.cache/` (git-ignored), and writes `bench/results/<date>-<host>.md` and `.json`. Commit the `.md` to keep a record.

## What it measures

**Storage.** For each model, `chungus bench --json` reports its size as raw files, with zstd on each chunk (the baseline any compressor gets), and with the full chungus pipeline (float transform, zstd, dedup within the model). It also breaks this down by tensor dtype, and reports chunking, encode and decode throughput.

**Xet.** For the same files, the report shows what Hugging Face's Xet storage sends: its compressed, deduplicated chunks, plus small files that live in git at full size. These numbers come from Xet's reconstruction API, which lists for each file the compressed byte ranges a client downloads, so nothing is modelled or reimplemented. The report opens with a *chungus vs Xet* table that sums this up per model and per update.

**Dedup between versions.** For a pair such as a base model and its fine-tune, `chungus bench base fine-tune` counts how much of the fine-tune is already in a store that holds the base, and what fetching it would transfer. This is the number that matters for a user who already has one version. The Xet columns show the same for a Xet client that kept every chunk of the first model, the most Xet's chunk cache could save. Xet only reports the size of a whole fetch range, so each chunk gets an equal share of its range.

**Download time.** Each run downloads into an empty cache, including an empty Xet chunk cache, with `huggingface_hub.snapshot_download`, timing only the download:

| Path | What it shows |
|---|---|
| huggingface_hub, Xet | Hugging Face with Xet (`hf_xet`, which the script installs). |
| huggingface_hub, no Xet | Hugging Face over plain HTTP, with `HF_HUB_DISABLE_XET=1`. |
| chungus hub, cold | A fresh `chungus hub` pulling from huggingface.co while it packs: the cost of going through chungus the first time. |
| chungus hub, warm | The same hub serving from its store: what the second machine in an office gets. |
| LAN peer (`--peer URL`) | A fresh offline hub filling from another hub on the LAN that already has the model. |
| swarm (`--swarm org/model`) | `chungus fetch --swarm` of a published registry name over the internet swarm. |

The hub rows also report the store's size on disk next to the files it holds.

## Options

```sh
# Your own choice of models: names from bench/run.py, org/name, or org/name@revision
uv run bench/run.py --compress Qwen/Qwen2.5-1.5B --pair Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-1.5B-Instruct

# Two revisions of one repo
uv run bench/run.py --compress --pair org/model@<old commit> org/model@main --no-download

# Models already on disk
uv run bench/run.py --local mine=/path/to/model --compress mine --no-download

# Also time a LAN hub that already has the models, and the swarm
uv run bench/run.py --peer http://192.168.1.20:8080 --swarm org/model
```

`--hf-endpoint` points everything at a mirror instead of huggingface.co. A model that fails to download or parse is listed under *Skipped* and the run carries on.

## What to expect

These follow from the format, and the first results so far agree with them:

- **BF16** weights shrink to about 70 to 75% of raw. The exponent split is what beats zstd alone here (zstd alone gets about 79%).
- **F32** weights save a little more than zstd alone (about 4 points in the run below).
- **F16** weights barely beat zstd: F16 gets the byte-plane split only, because its 5-bit exponent doesn't sit on a byte boundary. An exponent split for F16 was measured and doesn't help; see *F16 exponent split* below.
- **Quantized weights (GGUF Q4/Q8, FP8)** are close to incompressible. For these, chungus saves through dedup and peer transfer, not compression.
- **GGUF quantizations of one model** share more than their types suggest. Qwen2.5 0.5B Instruct's Q8_0 file shares about 151 of 676 MB with its Q4_K_M file, because Q4_K_M keeps `output.weight` in Q8_0 and both files carry the same 5.9 MB header (mostly the tokenizer). Most of the rest of Q8_0's savings on its own come from its tied embeddings: `output.weight` and `token_embd.weight` are the same 145 MB. Whole-file chunking already finds nearly all of this; splitting GGUF per tensor adds only the chunks at the edges of shared tensors, about 0.1% for this pair. The dedup table's *Whole files* column shows the difference for each GGUF pair.
- **Fine-tunes** that retrain every weight share almost nothing with their base, at the byte level. Dedup pays off when bytes really repeat: the same model in several repos or revisions, re-exports and re-shards of the same weights, frozen layers, shared embeddings and tokenizers, and one model used by many people on the same LAN or swarm.

## First results (sandbox, real weights, not LLMs)

The sandbox these were written in can't reach huggingface.co, so this first run used the real trained weights that ship inside PyPI packages (NudeNet, RapidOCR's PP-OCRv4 and Silero VAD, 29 MB in total), converted to safetensors in each dtype. They are small vision and audio networks, not LLMs, so treat them as a check that the tool works. The LLM numbers come from running the script on a machine that can reach Hugging Face.

Intel Xeon 2.1 GHz, 4 cores, Linux:

| Dtype | Raw | zstd | chungus |
|---|--:|--:|--:|
| BF16 | 14.6 MB | 79.0% | 74.7% |
| F16 | 14.6 MB | 92.3% | 92.0% |
| F32 | 29.3 MB | 92.5% | 88.8% |

| Have | Fetch | Already have |
|---|---|--:|
| RapidOCR 1.3.24 models | RapidOCR 1.4.4 models | 100% (same weights, new release) |
| Silero VAD ONNX | Silero VAD, opset 18 re-export | 17% |
| F32 copy | BF16 copy of the same weights | 0% (every byte differs) |

Throughput (MB/s of raw bytes): chunking about 1,600 on one core, hash and encode about 430, decode about 1,900 on four cores.

## F16 exponent split

Measured on 2026-09-30, and not worth building. BF16 and F32 chunks get an exponent split (sign and exponent in one byte plane), but F16 gets plain byte planes, whose high byte mixes sign, exponent and the top 2 mantissa bits. The candidate layout was three planes per chunk: sign and exponent (6 bits a byte), the top 2 mantissa bits packed four to a byte, and the low 8 mantissa bits, compressed with zstd level 3 in one frame as blobs are today. The bar for adding it (a blob version and a two-release rollout) was 2% fewer F16 bytes.

The sandbox can't reach Hugging Face, so the data was Whisper tiny.en's trained weights (sherpa-onnx's export, from GitHub), converted from F32 to F16: 37.7 million values, 75.5 MB, cut into 64 KB chunks.

| Encoding | Size | Share of raw |
|---|--:|--:|
| zstd | 69.6 MB | 92.2% |
| byte planes + zstd (today) | 69.4 MB | 92.0% |
| exponent split + zstd | 72.2 MB | 95.7% |
| best per chunk, today | 69.4 MB | 92.0% |
| best per chunk, with the exponent split | 69.4 MB | 92.0% |

The exponent split never wins a chunk, so it saves nothing. One thing did stand out: compressing each byte plane as its own zstd frame, rather than one frame over all planes, took the byte planes to 64.2 MB (85.1%), about 7% smaller, and the exponent split to 65.2 MB. zstd shares one entropy table across a block, and the near-random mantissa plane spoils it for the skewed exponent plane. That would be a blob format change for every float dtype, so it needs its own measurement on BF16 and F32 and on real F16 checkpoints (pythia-160m, Qwen2.5 0.5B's F16 GGUF) before it goes anywhere.
