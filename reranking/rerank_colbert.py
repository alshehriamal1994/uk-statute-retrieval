import argparse
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict
from transformers import AutoTokenizer, AutoModel
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

from evaluation.metrics import compute_metrics_at_ks

class ColBERTScorer:
    """
    ColBERT-style late interaction scorer using HuggingFace models.
    """
    def __init__(self, model_name: str, device: str = None, max_length: int = 256):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        print(f"Loading ColBERT model {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _encode(self, texts: List[str]):
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**enc)
            # last_hidden_state: (B, L, H)
            reps = outputs.last_hidden_state
        return reps, enc["attention_mask"]

    def score_query_docs(self, query: str, docs: List[str], batch_size: int = 16) -> List[float]:
        if not docs: return []
        
        # Encode Query
        q_emb, q_attn = self._encode([query]) 
        q_emb = F.normalize(q_emb, p=2, dim=-1)
        q_mask = q_attn.bool()

        scores = []
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i:i + batch_size]
            d_emb, d_attn = self._encode(batch_docs)
            d_emb = F.normalize(d_emb, p=2, dim=-1)
            d_mask = d_attn.bool()

            B, Ld, H = d_emb.shape
            _, Lq, _ = q_emb.shape
            
            # Expand query
            q_expanded = q_emb.expand(B, Lq, H)
            q_mask_exp = q_mask.expand(B, Lq)
            
            # Similarity
            sim = torch.matmul(q_expanded, d_emb.transpose(1, 2)) # (B, Lq, Ld)
            
            # Mask doc padding
            d_mask_exp = d_mask.unsqueeze(1).expand(B, Lq, Ld)
            sim = sim.masked_fill(~d_mask_exp, -1e9)
            
            # MaxSim
            max_per_q, _ = sim.max(dim=2)
            max_per_q = max_per_q.masked_fill(~q_mask_exp, 0.0)
            
            batch_scores = max_per_q.sum(dim=1)
            scores.extend(batch_scores.detach().cpu().tolist())
            
        return scores

def main():
    parser = argparse.ArgumentParser(description="Run ColBERT Reranker")
    parser.add_argument("--candidates", type=str, required=True, help="Candidates JSONL")
    parser.add_argument("--data_json", type=str, required=True, help="Corpus Data.json")
    parser.add_argument("--model_name", type=str, default="colbert-ir/colbertv2.0")
    parser.add_argument("--out_tsv", type=str, default="results_colbert.tsv")
    parser.add_argument("--max_docs", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    set_seed(42)

    data_path = Path(args.data_json)
    cand_path = Path(args.candidates)
    
    Path(args.out_tsv).parent.mkdir(parents=True, exist_ok=True)

    # Load corpus
    print(f"Loading corpus from {data_path}")
    with data_path.open("r", encoding="utf-8") as f:
        corpus = json.load(f)
    id2text = {item.get("metadata", {}).get("chunk_id", item.get("metadata", {}).get("doc_id")): item.get("content", "") for item in corpus}

    # Load Candidates
    cand_examples = []
    with cand_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): cand_examples.append(json.loads(line))

    scorer = ColBERTScorer(args.model_name, max_length=args.max_length)
    
    eval_for_metrics = []
    
    with open(args.out_tsv, "w", encoding="utf-8") as out_f:
        out_f.write("qid\tmodel\tchunk_id\trank\tscore\trel\n")

        for ex in tqdm(cand_examples, desc="ColBERT Reranking"):
            qid = ex["qid"]
            query = ex["query"]
            candidates = ex["candidates"][:args.max_docs]
            rel_map = {r["chunk_id"]: r.get("rel", 1) for r in ex.get("relevant", [])}
            
            cand_ids = [c["chunk_id"] for c in candidates]
            docs = [id2text.get(cid, "") for cid in cand_ids]
            
            valid_indices = [i for i, d in enumerate(docs) if d]
            if not valid_indices: continue
            
            valid_docs = [docs[i] for i in valid_indices]
            valid_ids = [cand_ids[i] for i in valid_indices]

            scores = scorer.score_query_docs(query, valid_docs)
            
            scored = list(zip(valid_ids, scores))
            scored.sort(key=lambda x: x[1], reverse=True)
            
            eval_for_metrics.append({
                "qid": qid,
                "relevant": ex.get("relevant", []),
                "reranked": scored
            })
            
            for rank, (cid, s) in enumerate(scored, start=1):
                rel = rel_map.get(cid, 0)
                out_f.write(f"{qid}\t{args.model_name}\t{cid}\t{rank}\t{s}\t{rel}\n")

    metrics = compute_metrics_at_ks(eval_for_metrics)
    print("=== ColBERT Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
