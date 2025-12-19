import json
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import random
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def simple_tokenize(text: str):
    """Simple tokenizer: lowercase + split on whitespace."""
    return text.lower().split()

def load_corpus(data_json: Path):
    print(f"Loading corpus from {data_json}")
    with data_json.open("r", encoding="utf-8") as f:
        corpus = json.load(f)
    print(f"Loaded {len(corpus)} chunks")

    chunk_ids = []
    docs = []
    for item in corpus:
        meta = item.get("metadata", {})
        chunk_id = meta.get("chunk_id") or meta.get("doc_id")
        text = item.get("content", "").strip()
        if not chunk_id or not text:
            continue
        chunk_ids.append(chunk_id)
        docs.append(text)

    print(f"Using {len(docs)} chunks after filtering")
    return chunk_ids, docs

def load_eval_queries(eval_jsonl: Path):
    print(f"Loading eval queries from {eval_jsonl}")
    eval_examples = []
    with eval_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            eval_examples.append(ex)
    print(f"Loaded {len(eval_examples)} eval queries")
    return eval_examples

def main():
    parser = argparse.ArgumentParser(description="Run BM25 retrieval on UK Statute corpus")
    parser.add_argument("--data_json", type=str, required=True, help="Path to Data.json corpus file")
    parser.add_argument("--eval_jsonl", type=str, required=True, help="Path to evaluation queries JSONL")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file for candidates")
    parser.add_argument("--k", type=int, default=100, help="Number of candidates to retrieve per query")
    args = parser.parse_args()

    set_seed(42)

    data_path = Path(args.data_json)
    eval_path = Path(args.eval_jsonl)
    out_path = Path(args.output_file)
    
    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load corpus and build index
    chunk_ids, docs = load_corpus(data_path)
    print("Tokenizing corpus...")
    tokenized_corpus = [simple_tokenize(d) for d in tqdm(docs, desc="Tokenizing")]
    
    print("Building BM25 Index...")
    bm25 = BM25Okapi(tokenized_corpus)

    # 2. Load queries
    eval_examples = load_eval_queries(eval_path)

    # 3. Retrieve
    print(f"Retrieving top-{args.k} candidates per query...")
    with out_path.open("w", encoding="utf-8") as out_f:
        for ex in tqdm(eval_examples, desc="BM25 Retrieval"):
            qid = ex.get("qid")
            query = ex.get("query")
            # Preserve existing relevant info if present
            relevant = ex.get("relevant", []) 
            if not relevant and "answers" in ex:
                 # Map old format "answers" -> "relevant"
                 relevant = []
                 for ans in ex["answers"]:
                     if ans.get("chunk_id"):
                         relevant.append({"chunk_id": ans["chunk_id"], "rel": ans.get("rel", 1)})

            q_tokens = simple_tokenize(query)
            scores = bm25.get_scores(q_tokens)
            
            # Efficient top-k
            if args.k >= len(scores):
                top_idx = np.argsort(scores)[::-1]
            else:
                top_idx = np.argpartition(scores, -args.k)[-args.k:]
                top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

            candidates = []
            for i in top_idx:
                candidates.append({
                    "chunk_id": chunk_ids[i],
                    "score": float(scores[i])
                })

            out_record = {
                "qid": qid,
                "query": query,
                "candidates": candidates,
                "relevant": relevant
            }
            out_f.write(json.dumps(out_record) + "\n")

    print(f"Saved candidates to {out_path}")

if __name__ == "__main__":
    main()
