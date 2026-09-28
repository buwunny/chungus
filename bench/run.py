# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub>=0.34"]
# ///
"""Benchmark chungus on real models: storage saved and download speed.

    uv run bench/run.py                 # the quick set, a few minutes on a laptop
    uv run bench/run.py --set full      # adds larger and quantized models
    uv run bench/run.py --help

Three measurements, all written to bench/results/<date>-<host>.{md,json}:

1. Compression per model: raw size vs zstd alone vs chungus's float transform + zstd,
   per tensor dtype, from `chungus bench --json`.
2. Dedup between versions: how much of a second model (a fine-tune, another revision)
   is already in a store that holds the first, i.e. what a user who has the first
   actually downloads.
3. Download time: huggingface_hub straight from the Hub vs through `chungus hub`,
   cold (the hub pulls from upstream while packing) and warm (served from its store),
   plus a LAN peer (--peer) or the internet swarm (--swarm) when given.

Models are downloaded once into bench/.cache (git-ignored). Timed downloads always go
into fresh temporary caches, so the cache never makes a timed run look faster.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "bench" / ".cache"
RESULTS = ROOT / "bench" / "results"

# Files worth measuring. Other formats in a repo (PyTorch .bin, ONNX, TF, Flax copies of
# the same weights) are skipped so every model is measured once, as safetensors or GGUF.
WEIGHTS = ["*.safetensors", "*.json", "*.txt", "*.model", "*.tiktoken"]


@dataclass
class Model:
    repo: str
    note: str
    revision: str | None = None
    patterns: list[str] = field(default_factory=lambda: list(WEIGHTS))

    @property
    def label(self) -> str:
        return self.repo + (f"@{self.revision[:12]}" if self.revision else "")


def gguf(repo: str, file: str, note: str) -> Model:
    return Model(repo, note, patterns=[file])


# Small to medium public, ungated models, so the quick set runs on a laptop.
MODELS = {
    "smollm2-135m": Model("HuggingFaceTB/SmolLM2-135M", "BF16 base"),
    "smollm2-135m-instruct": Model("HuggingFaceTB/SmolLM2-135M-Instruct", "BF16 fine-tune"),
    "gpt2": Model("openai-community/gpt2", "F32"),
    "minilm": Model("sentence-transformers/all-MiniLM-L6-v2", "F32 embedding model"),
    "pythia-160m": Model("EleutherAI/pythia-160m", "F16"),
    "qwen2.5-0.5b": Model("Qwen/Qwen2.5-0.5B", "BF16 base"),
    "qwen2.5-0.5b-instruct": Model("Qwen/Qwen2.5-0.5B-Instruct", "BF16 fine-tune"),
    "qwen2.5-0.5b-q4": gguf(
        "Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf", "GGUF Q4_K_M"
    ),
    "qwen2.5-0.5b-q8": gguf(
        "Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q8_0.gguf", "GGUF Q8_0"
    ),
    "qwen3-0.6b": Model("Qwen/Qwen3-0.6B", "BF16"),
    "qwen3-0.6b-fp8": Model("Qwen/Qwen3-0.6B-FP8", "FP8 (E4M3)"),
    "tinyllama-1.1b": Model("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "BF16, 2.2 GB"),
}

SETS = {
    "quick": {
        "compress": ["smollm2-135m-instruct", "gpt2", "minilm", "pythia-160m", "qwen2.5-0.5b-q4"],
        "pairs": [("smollm2-135m", "smollm2-135m-instruct")],
        "download": ["smollm2-135m-instruct"],
    },
    "full": {
        "compress": [
            "smollm2-135m-instruct",
            "gpt2",
            "minilm",
            "pythia-160m",
            "qwen2.5-0.5b-instruct",
            "qwen3-0.6b",
            "qwen3-0.6b-fp8",
            "qwen2.5-0.5b-q4",
            "qwen2.5-0.5b-q8",
            "tinyllama-1.1b",
        ],
        "pairs": [
            ("smollm2-135m", "smollm2-135m-instruct"),
            ("qwen2.5-0.5b", "qwen2.5-0.5b-instruct"),
            ("qwen3-0.6b", "qwen3-0.6b-fp8"),
        ],
        "download": ["smollm2-135m-instruct", "qwen2.5-0.5b-instruct"],
    },
}


def log(msg: str) -> None:
    print(f"[bench] {msg}", file=sys.stderr, flush=True)


def mb(n: float) -> str:
    return f"{n / 1e6:,.1f} MB"


def pct(part: float, whole: float) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "n/a"


def parse_model(spec: str) -> Model:
    """A MODELS key, `org/name` or `org/name@revision`."""
    if spec in MODELS:
        return MODELS[spec]
    repo, _, rev = spec.partition("@")
    return Model(repo, "", revision=rev or None)


def du(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.chungus = args.chungus or str(ROOT / "target" / "release" / "chungus")
        self.local: dict[str, Path] = {}
        for spec in args.local:
            name, _, path = spec.partition("=")
            self.local[name] = Path(path).resolve()
        self.errors: list[str] = []

    # --- helpers -------------------------------------------------------------------------

    def download(
        self, m: Model, cache_dir: Path, endpoint: str | None = None
    ) -> tuple[Path, float]:
        """snapshot_download in a child process, so HF_ENDPOINT is read fresh each time.
        Returns the snapshot folder and the seconds snapshot_download took."""
        env = dict(os.environ)
        env["HF_ENDPOINT"] = endpoint or self.args.hf_endpoint
        env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        env.pop("HF_HUB_OFFLINE", None)
        code = (
            "import json, sys, time\n"
            "from huggingface_hub import snapshot_download\n"
            "a = json.loads(sys.argv[1])\n"
            "t = time.perf_counter()\n"
            "p = snapshot_download(a['repo'], revision=a['rev'], cache_dir=a['cache'],"
            " allow_patterns=a['patterns'])\n"
            "print(time.perf_counter() - t)\n"
            "print(p)\n"
        )
        arg = json.dumps(
            {"repo": m.repo, "rev": m.revision, "cache": str(cache_dir), "patterns": m.patterns}
        )
        out = subprocess.run(
            [sys.executable, "-c", code, arg], env=env, check=True, capture_output=True, text=True
        )
        secs, path = out.stdout.strip().splitlines()[-2:]
        return Path(path), float(secs)

    def fetch(self, name: str) -> Path:
        """The model's files on disk, downloading them into bench/.cache once."""
        if name in self.local:
            return self.local[name]
        m = parse_model(name)
        log(f"getting {m.label}")
        return self.download(m, CACHE)[0]

    def bench(self, *paths: Path) -> dict:
        out = subprocess.run(
            [self.chungus, "bench", "--json", *map(str, paths)],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(out.stdout)

    def label(self, name: str) -> tuple[str, str]:
        if name in self.local:
            return name, "local files"
        m = parse_model(name)
        return m.label, m.note

    def attempt(self, what: str, f):
        try:
            return f()
        except subprocess.CalledProcessError as e:
            lines = (e.stderr or "").strip().splitlines()
            msg = f"{what}: {lines[-1] if lines else e}"
        except Exception as e:  # noqa: BLE001 - one broken model shouldn't stop the run
            msg = f"{what}: {e}"
        log(f"skipped {msg}")
        self.errors.append(msg)
        return None

    # --- 1. compression ------------------------------------------------------------------

    def compression(self, names: list[str]) -> list[dict]:
        rows = []
        for name in names:

            def one(name=name):
                path = self.fetch(name)
                log(f"bench {name}")
                r = self.bench(path)
                label, note = self.label(name)
                return {"model": label, "note": note, **r}

            if (r := self.attempt(name, one)) is not None:
                rows.append(r)
        return rows

    # --- 2. dedup ------------------------------------------------------------------------

    def dedup(self, pairs: list[tuple[str, str]]) -> list[dict]:
        rows = []
        for a, b in pairs:

            def one(a=a, b=b):
                pa, pb = self.fetch(a), self.fetch(b)
                log(f"bench {a} then {b}")
                r = self.bench(pa, pb)
                second = r["inputs"][1]
                alone = self.bench(pb)
                return {
                    "first": self.label(a)[0],
                    "second": self.label(b)[0],
                    "raw_bytes": second["raw_bytes"],
                    "new_raw_bytes": second["new_raw_bytes"],
                    "fetch_bytes": second["new_stored_bytes"],
                    "fetch_bytes_alone": alone["dedup_bytes"],
                }

            if (r := self.attempt(f"{a} -> {b}", one)) is not None:
                rows.append(r)
        return rows

    # --- 3. download speed ---------------------------------------------------------------

    def start_hub(self, store: Path, extra: list[str]) -> tuple[subprocess.Popen, str]:
        port = free_port()
        cmd = [
            self.chungus,
            "hub",
            "--store",
            str(store),
            "--port",
            str(port),
            "--upstream",
            self.args.hf_endpoint,
            *extra,
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        url = f"http://127.0.0.1:{port}"
        for _ in range(100):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    return proc, url
            except OSError:
                if proc.poll() is not None:
                    raise RuntimeError(f"chungus hub exited with {proc.returncode}")
                time.sleep(0.1)
        proc.kill()
        raise RuntimeError("chungus hub did not start")

    def timed_download(self, m: Model, endpoint: str | None = None) -> dict:
        with tempfile.TemporaryDirectory(prefix="chungus-bench-") as tmp:
            _, secs = self.download(m, Path(tmp), endpoint)
            # The snapshot folder is symlinks into blobs/, so measure the whole cache.
            return {"secs": secs, "bytes": du(Path(tmp))}

    def downloads(self, names: list[str]) -> list[dict]:
        rows = []
        for name in names:
            m = parse_model(name)

            def one(m=m):
                row = {"model": m.label, "runs": {}}
                log(f"download {m.label} from {self.args.hf_endpoint}")
                row["runs"]["huggingface_hub, direct"] = self.timed_download(m)
                with tempfile.TemporaryDirectory(prefix="chungus-hub-") as tmp:
                    store = Path(tmp) / "store"
                    hub, url = self.start_hub(store, ["--no-mdns"])
                    try:
                        log(f"download {m.label} through a cold chungus hub")
                        row["runs"]["chungus hub, cold"] = self.timed_download(m, url)
                        row["store_bytes"] = du(store)
                        log(f"download {m.label} through the same hub, warm")
                        row["runs"]["chungus hub, warm"] = self.timed_download(m, url)
                    finally:
                        hub.terminate()
                        hub.wait()
                for peer in self.args.peer:
                    with tempfile.TemporaryDirectory(prefix="chungus-hub-") as tmp:
                        hub, url = self.start_hub(
                            Path(tmp) / "store", ["--no-mdns", "--offline", "--peer", peer]
                        )
                        try:
                            log(f"download {m.label} from LAN peer {peer}")
                            row["runs"][f"LAN peer {peer}"] = self.timed_download(m, url)
                        finally:
                            hub.terminate()
                            hub.wait()
                return row

            if (r := self.attempt(f"download {name}", one)) is not None:
                rows.append(r)

        for model in self.args.swarm:

            def swarm(model=model):
                with tempfile.TemporaryDirectory(prefix="chungus-swarm-") as tmp:
                    log(f"fetch {model} over the swarm")
                    t = time.perf_counter()
                    subprocess.run(
                        [
                            self.chungus,
                            "fetch",
                            model,
                            "--swarm",
                            "--store",
                            f"{tmp}/store",
                            "-o",
                            f"{tmp}/out",
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    secs = time.perf_counter() - t
                    return {
                        "model": model,
                        "runs": {
                            "chungus fetch --swarm": {"secs": secs, "bytes": du(Path(tmp) / "out")}
                        },
                        "store_bytes": du(Path(tmp) / "store"),
                    }

            if (r := self.attempt(f"swarm {model}", swarm)) is not None:
                rows.append(r)
        return rows


def report(
    meta: dict, comp: list[dict], dedup: list[dict], dl: list[dict], errors: list[str]
) -> str:
    out = [f"# chungus benchmark, {meta['date']}", ""]
    out.append(
        f"{meta['host']}: {meta['cpu']}, {meta['cores']} cores, {meta['os']}. "
        f"chungus {meta['commit']}. Upstream {meta['endpoint']}."
    )
    if comp:
        out += [
            "",
            "## Storage",
            "",
            "Size as a share of the raw files (lower is better). *zstd* compresses each chunk "
            "on its own; *chungus* adds the float transform and dedup within the model.",
            "",
            "| Model | Kind | Raw | zstd | chungus | Saved | Decode |",
            "|---|---|--:|--:|--:|--:|--:|",
        ]
        for r in comp:
            raw = r["raw_bytes"]
            out.append(
                f"| {r['model']} | {r['note']} | {mb(raw)} | {pct(r['zstd_bytes'], raw)} | "
                f"{pct(r['dedup_bytes'], raw)} | {mb(raw - r['dedup_bytes'])} | "
                f"{raw / 1e6 / max(r['decode_secs'], 1e-9):,.0f} MB/s |"
            )
        out += [
            "",
            "By tensor dtype, over every model above:",
            "",
            "| Dtype | Raw | zstd | chungus |",
            "|---|--:|--:|--:|",
        ]
        totals: dict[str, list[int]] = {}
        for r in comp:
            for k, d in r["by_dtype"].items():
                t = totals.setdefault(k, [0, 0, 0])
                for i, key in enumerate(["raw_bytes", "zstd_bytes", "encoded_bytes"]):
                    t[i] += d[key]
        for k, (raw, z, e) in sorted(totals.items(), key=lambda kv: -kv[1][0]):
            out.append(f"| {k} | {mb(raw)} | {pct(z, raw)} | {pct(e, raw)} |")
        out += [
            "",
            "`raw` is everything without a float layout chungus transforms: headers, "
            "configs, integer, FP8 and quantized (GGUF) tensors.",
        ]
        chunk = sum(r["raw_bytes"] for r in comp) / 1e6
        out += [
            "",
            "Throughput over all models (MB/s of raw bytes): "
            f"chunking {chunk / sum(r['chunk_secs'] for r in comp):,.0f} (one core), "
            f"hash + encode {chunk / sum(r['encode_secs'] for r in comp):,.0f}, "
            f"decode {chunk / sum(r['decode_secs'] for r in comp):,.0f} (all cores).",
        ]
    if dedup:
        out += [
            "",
            "## Dedup between versions",
            "",
            "What fetching the second model costs when the first is already in the store.",
            "",
            "| Have | Fetch | Size | Already have | Transfer | Without dedup |",
            "|---|---|--:|--:|--:|--:|",
        ]
        for r in dedup:
            out.append(
                f"| {r['first']} | {r['second']} | {mb(r['raw_bytes'])} | "
                f"{pct(r['raw_bytes'] - r['new_raw_bytes'], r['raw_bytes'])} | "
                f"{mb(r['fetch_bytes'])} | {mb(r['fetch_bytes_alone'])} |"
            )
    if dl:
        out += [
            "",
            "## Download time",
            "",
            "Each run downloads into an empty cache. *Cold* is a fresh hub pulling from "
            "upstream while it packs; *warm* is the same hub serving from its store.",
            "",
            "| Model | Path | Time | Speed |",
            "|---|---|--:|--:|",
        ]
        for r in dl:
            for how, run in r["runs"].items():
                out.append(
                    f"| {r['model']} | {how} | {run['secs']:.1f} s | "
                    f"{run['bytes'] / 1e6 / max(run['secs'], 1e-9):,.0f} MB/s |"
                )
            if "store_bytes" in r:
                size = max(run["bytes"] for run in r["runs"].values())
                out.append(
                    f"| | chungus store on disk | {mb(r['store_bytes'])} | "
                    f"{pct(r['store_bytes'], size)} of the files |"
                )
    if errors:
        out += ["", "## Skipped", ""] + [f"- {e}" for e in errors]
    return "\n".join(out) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--set", choices=SETS, default="quick", help="which models to run (default quick)"
    )
    p.add_argument(
        "--compress",
        nargs="*",
        metavar="MODEL",
        help="models for the storage table, overriding the set: a name from the list "
        "in this file, org/name, org/name@revision, or a --local name",
    )
    p.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("HAVE", "FETCH"),
        help="a dedup pair, overriding the set's pairs; repeatable",
    )
    p.add_argument("--download", nargs="*", metavar="MODEL", help="models to time downloads of")
    p.add_argument(
        "--local",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="a model directory already on disk, usable by NAME in the lists",
    )
    p.add_argument(
        "--peer",
        action="append",
        default=[],
        metavar="URL",
        help="also time downloads from this LAN hub (it must already have the models)",
    )
    p.add_argument(
        "--swarm",
        action="append",
        default=[],
        metavar="NAME",
        help="also time `chungus fetch --swarm NAME` (a published registry name)",
    )
    p.add_argument("--no-download", action="store_true", help="skip the download timings")
    p.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    p.add_argument("--chungus", help="chungus binary (default: build target/release/chungus)")
    p.add_argument("--out", type=Path, default=RESULTS, help="results directory")
    args = p.parse_args()

    if not args.chungus:
        log("building chungus (release)")
        subprocess.run(["cargo", "build", "--release", "--quiet"], cwd=ROOT, check=True)

    s = SETS[args.set]
    compress = args.compress if args.compress is not None else s["compress"]
    pairs = [tuple(x) for x in args.pair] if args.pair else s["pairs"]
    download = (
        [] if args.no_download else (args.download if args.download is not None else s["download"])
    )

    run = Runner(args)
    comp = run.compression(compress)
    dedup = run.dedup(pairs)
    dl = run.downloads(download) if not args.no_download else []

    commit = (
        subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()
        or "unknown"
    )
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    if sys.platform == "darwin":
        cpu = (
            subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            ).stdout.strip()
            or cpu
        )
    meta = {
        "date": dt.date.today().isoformat(),
        "host": platform.node().split(".")[0],
        "cpu": cpu,
        "cores": os.cpu_count(),
        "os": f"{platform.system()} {platform.release()}",
        "commit": commit,
        "endpoint": args.hf_endpoint,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / f"{meta['date']}-{meta['host']}"
    stem.with_suffix(".json").write_text(
        json.dumps(
            {
                "meta": meta,
                "compression": comp,
                "dedup": dedup,
                "downloads": dl,
                "skipped": run.errors,
            },
            indent=2,
        )
    )
    md = report(meta, comp, dedup, dl, run.errors)
    stem.with_suffix(".md").write_text(md)
    print(md)
    log(f"wrote {stem}.md and {stem}.json")


if __name__ == "__main__":
    main()
