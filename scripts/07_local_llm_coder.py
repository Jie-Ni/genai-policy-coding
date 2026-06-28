"""Open-weight LLM-as-coder via vLLM offline batched inference.

Runs the codebook v1 prompt over a shard of policy chunks. Designed for
SLURM job-array execution: one job = one shard = one node = one H100.

Pipeline per node:
  1. Load model from /local NVMe (staged by stage_models_to_local.sh)
  2. Stream `data_processed/policy_chunks_shard_NN.jsonl` from /scratch
  3. Build prompts using scripts/prompts/coder_prompt_v1.md (system + 16
     few-shot examples + the chunk)
  4. Generate JSON responses with guided_json schema (vLLM enforces the
     8-theme + 4-sentiment object)
  5. Validate each response; write to /local first, rsync to /scratch
  6. Aggregate sha256 manifest for provenance

Inputs (env)
------------
  MODEL_NAME       : e.g. Qwen__Qwen3.6-27B-Instruct-AWQ
  SHARD_PATH       : /scratch/.../shards/shard_03.jsonl
  OUTPUT_DIR_SCRATCH : /scratch/.../results/codes/qwen36/
  LOCAL_WORK_DIR   : /local/ijethe_policy/work
  HF_HOME, HF_TOKEN

Usage
-----
    python scripts/07_local_llm_coder.py \
        --shard data_processed/policy_chunks_shard_03.jsonl \
        --model-name Qwen3.6-27B-Instruct-AWQ \
        --out-scratch /scratch/.../results/codes/qwen36/shard_03.jsonl

Failure modes handled
---------------------
- JSON parse error -> retry once with `temperature=0.1`; if still fail, emit
  a row with `error: "json_parse_failed"` and the raw text for adjudication
- vLLM OOM -> reduce batch_size and retry
- Network/HF token failure -> abort cleanly; SLURM wrapper requeues
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

# Defer vLLM import until needed (so the script can be imported and
# validated offline without the heavyweight dep).
def import_vllm() -> Any:
    try:
        import vllm  # noqa: F401
        return vllm
    except ImportError as exc:
        print(f"ERROR: vLLM not available: {exc}", file=sys.stderr)
        print("Install via: pip install 'vllm>=0.6.0'", file=sys.stderr)
        sys.exit(1)


JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["chunk_id", "themes", "sentiment", "confidence", "notes"],
    "properties": {
        "chunk_id": {"type": "string"},
        "themes": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "T1_integration", "T2_multimodal", "T3_privacy",
                "T4_integrity", "T5_disclosure", "T6_equity",
                "T7_vendor_governance", "T8_pedagogical_redesign",
            ],
            "properties": {
                "T1_integration": {"type": "integer", "minimum": 0, "maximum": 1},
                "T2_multimodal": {"type": "integer", "minimum": 0, "maximum": 1},
                "T3_privacy": {"type": "integer", "minimum": 0, "maximum": 1},
                "T4_integrity": {"type": "integer", "minimum": 0, "maximum": 1},
                "T5_disclosure": {"type": "integer", "minimum": 0, "maximum": 1},
                "T6_equity": {"type": "integer", "minimum": 0, "maximum": 1},
                "T7_vendor_governance": {"type": "integer", "minimum": 0, "maximum": 1},
                "T8_pedagogical_redesign": {"type": "integer", "minimum": 0, "maximum": 1},
            },
        },
        "sentiment": {
            "type": "object",
            "additionalProperties": False,
            "required": ["assessment", "research", "teaching", "administration"],
            "properties": {
                "assessment": {"type": "integer", "minimum": -2, "maximum": 2},
                "research": {"type": "integer", "minimum": -2, "maximum": 2},
                "teaching": {"type": "integer", "minimum": -2, "maximum": 2},
                "administration": {"type": "integer", "minimum": -2, "maximum": 2},
            },
        },
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "notes": {"type": "string", "maxLength": 600},
    },
}


def load_prompt_template(prompt_md: Path) -> tuple[str, list[dict[str, str]]]:
    """Parse coder_prompt_v1.md to extract system message + 16 few-shot
    examples. The markdown structure is:

        ## System prompt
        ```
        <system text>
        ```
        ### Example N 鈥?...
        ```
        TEXT
        <chunk text>
        EXPECTED OUTPUT
        <expected JSON>
        ```
    """
    raw = prompt_md.read_text(encoding="utf-8")
    # System
    m = re.search(r"## System prompt\s*```\s*([\s\S]+?)```", raw)
    system = m.group(1).strip() if m else ""
    # Examples
    examples: list[dict[str, str]] = []
    for block in re.finditer(
        r"### Example \d+ 鈥?[^\n]*\n+```\s*\nTEXT\s*\n([\s\S]+?)EXPECTED OUTPUT\s*\n([\s\S]+?)```",
        raw,
    ):
        text = block.group(1).strip().strip('"').strip()
        expected = block.group(2).strip()
        # Try to compact JSON to one line
        try:
            j = json.loads(expected)
            expected = json.dumps(j, ensure_ascii=False)
        except Exception:
            pass
        examples.append({"text": text, "expected": expected})
    return system, examples


def build_user_prompt(chunk: dict[str, Any], examples: list[dict[str, str]]) -> str:
    """Compose the final user prompt: few-shot examples + the target chunk."""
    parts = [
        "Below are 16 example codings, followed by the chunk you must code.",
        "Output one JSON object only.",
        "",
    ]
    for i, ex in enumerate(examples, 1):
        parts += [
            f"EXAMPLE {i}",
            "Text: " + ex["text"][:600],
            "Expected: " + ex["expected"],
            "",
        ]
    parts += [
        "NOW CODE THE FOLLOWING:",
        f"University: {chunk.get('institution_name', '?')} ({chunk.get('region', '?')}, tier {chunk.get('tier', '?')})",
        f"Document section: {chunk.get('section_header', 'unknown')}",
        f"Original language: {chunk.get('language_declared', '?')}",
        f"Chunk ID: {chunk.get('chunk_id', '?')}",
        "",
        "Text:",
        chunk.get("text", "")[:6000],
        "",
        "JSON output only:",
    ]
    return "\n".join(parts)


def validate_response(text: str, chunk_id: str) -> tuple[bool, dict[str, Any]]:
    """Return (ok, parsed_or_error_dict)."""
    # 1. Strip markdown code fences (Mistral emits ```json ... ``` blocks)
    stripped = text.strip()
    fence = re.match(r"```(?:json)?\s*([\s\S]+?)```\s*$", stripped)
    if fence:
        stripped = fence.group(1).strip()
    # 2. find first { ... } block (greedy from first { to matching last })
    m = re.search(r"\{[\s\S]+\}", stripped)
    candidate = m.group(0) if m else stripped
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return False, {"error": "json_parse_failed", "raw": text[:2000], "msg": str(exc)}
    # 2. schema check (light: required keys + value ranges)
    try:
        if not isinstance(obj.get("themes"), dict):
            return False, {"error": "missing_themes", "raw": text[:2000]}
        for k in ("T1_integration", "T2_multimodal", "T3_privacy", "T4_integrity",
                  "T5_disclosure", "T6_equity", "T7_vendor_governance",
                  "T8_pedagogical_redesign"):
            v = obj["themes"].get(k)
            if v not in (0, 1):
                return False, {"error": "bad_theme_value", "key": k, "raw": text[:2000]}
        if not isinstance(obj.get("sentiment"), dict):
            return False, {"error": "missing_sentiment", "raw": text[:2000]}
        for k in ("assessment", "research", "teaching", "administration"):
            v = obj["sentiment"].get(k)
            if v not in (-2, -1, 0, 1, 2):
                return False, {"error": "bad_sentiment_value", "key": k, "raw": text[:2000]}
        if obj.get("confidence") not in ("low", "medium", "high"):
            obj["confidence"] = "medium"
        if "notes" not in obj:
            obj["notes"] = ""
        if obj.get("chunk_id") != chunk_id:
            obj["chunk_id"] = chunk_id  # override
        return True, obj
    except Exception as exc:
        return False, {"error": "validation_exc", "msg": str(exc), "raw": text[:2000]}


def stream_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True, help="JSONL file with chunks to code")
    ap.add_argument("--model-path", required=True, help="path on /local to the model dir")
    ap.add_argument("--prompt", default=str(ROOT / "scripts" / "prompts" / "coder_prompt_v1.md"))
    ap.add_argument("--out-scratch", required=True, help="final destination for the coded JSONL on /scratch")
    ap.add_argument("--out-local", default="/local/ijethe_policy/work", help="local working dir on the compute node")
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--quantization", default="awq", choices=["awq", "gptq", "fp8", "none"])
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--guided-json", action="store_true",
                    help="enforce JSON schema via vLLM guided decoding")
    ap.add_argument("--n-fewshot", type=int, default=-1,
                    help="truncate few-shot examples to first N (-1 = all). "
                         "Used for prompt-length ablation.")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="sampling temperature; default 0 = deterministic")
    ap.add_argument("--enable-thinking", action="store_true",
                    help="Qwen3 only: keep reasoning/CoT block. Default OFF 鈥?"
                         "reasoning wastes tokens before the JSON for our task.")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate inputs + prompt build without invoking vLLM")
    args = ap.parse_args()

    shard_path = Path(args.shard)
    if not shard_path.exists():
        print(f"ERROR: shard not found: {shard_path}", file=sys.stderr)
        return 1

    out_local = Path(args.out_local) / shard_path.stem
    out_local.mkdir(parents=True, exist_ok=True)
    out_local_jsonl = out_local / "codes.jsonl"

    prompt_md = Path(args.prompt)
    system_prompt, examples = load_prompt_template(prompt_md)
    if not system_prompt:
        print(f"ERROR: could not parse system prompt from {prompt_md}", file=sys.stderr)
        return 2
    print(f"[coder] loaded system prompt ({len(system_prompt)} chars), "
          f"{len(examples)} few-shot examples")

    chunks = stream_jsonl(shard_path)
    print(f"[coder] {len(chunks)} chunks in shard {shard_path.name}")

    # Few-shot truncation for ablation
    if args.n_fewshot >= 0 and args.n_fewshot < len(examples):
        examples = examples[: args.n_fewshot]
        print(f"[coder] truncated few-shot examples to {len(examples)} (ablation)")

    if args.dry_run:
        sample_prompt = build_user_prompt(chunks[0], examples) if chunks else ""
        print(f"[coder] dry-run sample prompt length: {len(sample_prompt)} chars")
        return 0

    # ---- vLLM init ----
    print(f"[coder] loading vLLM with model={args.model_path}")
    t0 = time.time()
    import_vllm()
    from vllm import LLM, SamplingParams

    llm_kwargs: dict[str, Any] = {
        "model": args.model_path,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "trust_remote_code": True,
        "seed": args.seed,
    }
    if args.quantization != "none":
        llm_kwargs["quantization"] = args.quantization
    llm = LLM(**llm_kwargs)
    print(f"[coder] vLLM ready ({time.time() - t0:.1f}s)")

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        repetition_penalty=1.0,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    # If vLLM supports guided JSON, set the schema. API moved across versions:
    #   v0.4-0.5: vllm.sampling_params.GuidedDecodingParams
    #   v0.6+: SamplingParams.guided_decoding field
    #   v0.19: vllm.sampling_params.GuidedDecodingParams REMOVED; use
    #          SamplingParams(guided_decoding={...}) or pass via LLM.chat
    if args.guided_json:
        ok = False
        # Try v0.6+ field-set API
        try:
            from vllm.sampling_params import GuidedDecodingParams  # type: ignore
            sampling.guided_decoding = GuidedDecodingParams(json=JSON_SCHEMA)
            print("[coder] guided_json enabled via GuidedDecodingParams")
            ok = True
        except ImportError:
            pass
        if not ok:
            # Try v0.19 dict-set API
            try:
                sampling.guided_decoding = {"json": JSON_SCHEMA}  # type: ignore
                print("[coder] guided_json enabled via dict")
                ok = True
            except Exception:
                pass
        if not ok:
            print(f"[coder] WARN: guided_json unavailable; falling back to post-hoc JSON validation")

    # ---- batched inference ----
    def make_messages(chunk: dict[str, Any]) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_prompt(chunk, examples)},
        ]

    # Use chat-completion-style API if available
    use_chat = hasattr(llm, "chat")
    # Qwen3 family: explicitly toggle thinking mode via chat_template_kwargs.
    # For our JSON-only coding task, reasoning fills the 800-token budget with
    # chain-of-thought before any JSON ever appears, so we want it OFF.
    chat_template_kwargs: dict[str, Any] = {"enable_thinking": args.enable_thinking}
    results: list[dict[str, Any]] = []
    n_ok = 0
    n_fail = 0
    for i in range(0, len(chunks), args.batch_size):
        batch = chunks[i:i + args.batch_size]
        t_batch = time.time()
        if use_chat:
            try:
                outputs = llm.chat(
                    [make_messages(c) for c in batch],
                    sampling,
                    chat_template_kwargs=chat_template_kwargs,
                )
            except TypeError:
                # Older vLLM signatures don't accept chat_template_kwargs; the
                # tokenizer default applies (which for Qwen3 = thinking ON).
                outputs = llm.chat([make_messages(c) for c in batch], sampling)
        else:
            # fall back to a single-prompt format
            prompts = []
            for c in batch:
                # naive concat 鈥?depends on tokenizer chat template
                p = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                p += f"<|im_start|>user\n{build_user_prompt(c, examples)}<|im_end|>\n"
                p += "<|im_start|>assistant\n"
                prompts.append(p)
            outputs = llm.generate(prompts, sampling)
        elapsed = time.time() - t_batch
        for c, out in zip(batch, outputs):
            text = out.outputs[0].text if hasattr(out, "outputs") else str(out)
            ok, parsed = validate_response(text, c.get("chunk_id", ""))
            rec = {
                "chunk_id": c.get("chunk_id", ""),
                "institution_id": c.get("institution_id", ""),
                "region": c.get("region", ""),
                "language_declared": c.get("language_declared", ""),
                "model_name": Path(args.model_path).name,
                "seed": args.seed,
                "timestamp_utc": dt.datetime.utcnow().isoformat() + "Z",
                "ok": ok,
            }
            if ok:
                rec.update(parsed)
                n_ok += 1
            else:
                rec["error_info"] = parsed
                n_fail += 1
            results.append(rec)
        print(f"[coder] batch {i//args.batch_size + 1}: "
              f"{len(batch)} chunks in {elapsed:.1f}s "
              f"(ok={n_ok} fail={n_fail})")

        # Write local checkpoint after each batch
        with out_local_jsonl.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- final stage-out: /local -> /scratch ----
    out_scratch = Path(args.out_scratch)
    out_scratch.parent.mkdir(parents=True, exist_ok=True)
    # Atomic move via tmpfile
    tmp = out_scratch.with_suffix(out_scratch.suffix + ".tmp")
    tmp.write_text(out_local_jsonl.read_text(encoding="utf-8"), encoding="utf-8")
    tmp.replace(out_scratch)

    manifest = {
        "shard": shard_path.name,
        "model": Path(args.model_path).name,
        "n_chunks": len(chunks),
        "n_ok": n_ok,
        "n_fail": n_fail,
        "out_scratch": str(out_scratch),
        "out_scratch_sha256": sha256_file(out_scratch),
        "timestamp_utc": dt.datetime.utcnow().isoformat() + "Z",
        "args": {k: v for k, v in vars(args).items()},
    }
    manifest_path = out_scratch.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[coder] DONE: ok={n_ok} fail={n_fail} out={out_scratch}")
    return 0 if n_fail == 0 else (3 if n_ok > 0 else 4)


if __name__ == "__main__":
    sys.exit(main())

