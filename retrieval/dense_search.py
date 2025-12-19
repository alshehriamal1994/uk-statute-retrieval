import json
import argparse
from pathlib import Path
import numpy as np
import faiss
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_corpus(path: Path):
    print(f"Loading corpus from {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    chunk_ids = []
    texts = []
    
    for item in data:
        meta = item.get("metadata", {})
        cid = meta.get("chunk_id") or meta.get("doc_id")
        text = item.get("content", "").strip()
        if cid and text:
            chunk_ids.append(cid)
            texts.append(text)

    print(f"Loaded {len(chunk_ids)} chunks")
    return chunk_ids, texts

def load_eval_queries(path: Path):
    print(f"Loading eval queries from {path}")
    eval_examples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            eval_examples.append(ex)
    print(f"Loaded {len(eval_examples)} eval queries")
    return eval_examples

def main():
    parser = argparse.ArgumentParser(description="Run Dense Retrieval (e.g. MPNet) on UK Statute Corpus")
    parser.add_argument("--data_json", type=str, required=True, help="Path to Data.json corpus file")
    parser.add_argument("--eval_jsonl", type=str, required=True, help="Path to evaluation queries JSONL")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file for candidates")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-mpnet-base-v2", help="HuggingFace model name")
    parser.add_argument("--k", type=int, default=100, help="Number of candidates to retrieve")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size")
    args = parser.parse_args()

    set_seed(42)

    data_path = Path(args.data_json)
    eval_path = Path(args.eval_jsonl)
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load Corpus
    chunk_ids, texts = load_corpus(data_path)

    # 2. Build FAISS Index
    print(f"Loading model: {args.model_name}")
    model = SentenceTransformer(args.model_name)
    
    print("Encoding corpus...")
    embs = model.encode(
        texts, 
        batch_size=args.batch_size, 
        show_progress_bar=True, 
        convert_to_numpy=True, 
        normalize_embeddings=True
    ).astype("float32")
    
    print(f"Building FAISS index (shape: {embs.shape})...")
    dimension = embs.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embs)

    # 3. Retrieve
    eval_examples = load_eval_queries(eval_path)
    
    print(f"Retrieving top-{args.k} candidates...")
    with out_path.open("w", encoding="utf-8") as out_f:
        for ex in tqdm(eval_examples, desc="Dense Retrieval"):
            qid = ex.get("qid")
            query = ex.get("query")
            
            # Encode Query
            q_emb = model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            ).astype("float32")
            
            # Search
            scores, idxs = index.search(q_emb, args.k)
            scores = scores[0]
            idxs = idxs[0]
            
            candidates = []
            for score, idx in zip(scores, idxs):
                candidates.append({
                    "chunk_id": chunk_ids[idx],
                    "score": float(score)
                })

            # Handle relevant data
            relevant = ex.get("relevant", [])
            if not relevant and "answers" in ex:
                 for ans in ex["answers"]:
                     if ans.get("chunk_id"):
                         relevant.append({"chunk_id": ans["chunk_id"], "rel": ans.get("rel", 1)})
            
            record = {
                "qid": qid,
                "query": query,
                "candidates": candidates,
                "relevant": relevant
            }
            out_f.write(json.dumps(record) + "\n")

    print(f"Saved candidates to {out_path}")

if __name__ == "__main__":
    main()
