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
- **F16** weights barely beat zstd: F16 gets the byte-plane split only, because its 5-bit exponent doesn't sit on a byte boundary. An F16 exponent transform is an open improvement.
- **Quantized weights (GGUF Q4/Q8, FP8)** are close to incompressible. For these, chungus saves through dedup and peer transfer, not compression.
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
