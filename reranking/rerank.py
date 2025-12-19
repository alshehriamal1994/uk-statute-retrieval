import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from tqdm import tqdm

from sentence_transformers import CrossEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
)

from evaluation.metrics import compute_metrics_at_ks
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ===================== monoT5 scorer =====================

class MonoT5Scorer:
    def __init__(self, model_name: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading monoT5 model {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        true_ids = self.tokenizer.encode("true", add_special_tokens=False)
        self.true_id = true_ids[0]

    def score_pairs(self, queries: List[str], docs: List[str], batch_size: int = 8) -> List[float]:
        scores = []
        for i in range(0, len(queries), batch_size):
            batch_q = queries[i:i + batch_size]
            batch_d = docs[i:i + batch_size]
            
            inputs = [f"Query: {q} Document: {d} Relevant:" for q, d in zip(batch_q, batch_d)]
            
            enc = self.tokenizer(inputs, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            decoder_input_ids = torch.full((enc.input_ids.size(0), 1), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, decoder_input_ids=decoder_input_ids)
                logits = outputs.logits[:, 0, :]
                probs = torch.softmax(logits, dim=-1)
                scores.extend(probs[:, self.true_id].detach().cpu().tolist())
        return scores

# ===================== HF seq-class scorer =====================

class HFSeqClsScorer:
    def __init__(self, model_name: str, device: str = None, max_length: int = 512):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if max_length > 8192: max_length = 8192
        self.max_length = max_length
        print(f"Loading HF seq-class model {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.model_max_length = self.max_length
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def score_pairs(self, queries: List[str], docs: List[str], batch_size: int = 8) -> List[float]:
        scores = []
        for i in range(0, len(queries), batch_size):
            batch_q = queries[i:i + batch_size]
            batch_d = docs[i:i + batch_size]
            
            inputs = self.tokenizer(batch_q, batch_d, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                if logits.ndim == 2 and logits.size(1) > 1:
                    batch_scores = logits[:, -1].detach().cpu().tolist()
                else:
                    batch_scores = logits.squeeze(-1).detach().cpu().tolist()
            
            if isinstance(batch_scores, float): batch_scores = [batch_scores]
            scores.extend(batch_scores)
        return scores

# ===================== Main =====================

def main():
    parser = argparse.ArgumentParser(description="Run Reranking Models")
    parser.add_argument("--candidates", type=str, required=True, help="Candidates JSONL file")
    parser.add_argument("--data_json", type=str, required=True, help="Corpus Data.json file")
    parser.add_argument("--model_type", type=str, choices=["cross-encoder", "monot5", "hf-seqcls", "cohere", "voyage"], required=True)
    parser.add_argument("--model_name", type=str, required=True, help="Model name (HF path or API model)")
    parser.add_argument("--out_tsv", type=str, default="results.tsv", help="Output TSV file")
    parser.add_argument("--max_docs", type=int, default=100, help="Top-N candidates to rerank")
    args = parser.parse_args()

    set_seed(42)

    data_path = Path(args.data_json)
    cand_path = Path(args.candidates)
    
    # Ensure output dir
    Path(args.out_tsv).parent.mkdir(parents=True, exist_ok=True)

    # 1. Load Corpus
    print(f"Loading corpus from {data_path}")
    with data_path.open("r", encoding="utf-8") as f:
        corpus = json.load(f)
    id2text = {item.get("metadata", {}).get("chunk_id", item.get("metadata", {}).get("doc_id")): item.get("content", "") for item in corpus}

    # 2. Load Candidates
    print(f"Loading candidates from {cand_path}")
    cand_examples = []
    with cand_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): cand_examples.append(json.loads(line))

    # 3. Initialize Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.model_type == "cross-encoder":
        print(f"Loading CrossEncoder: {args.model_name}")
        model = CrossEncoder(args.model_name, device=device)
        def score_func(q, d): 
            pairs = list(zip([q]*len(d), d))
            return model.predict(pairs).tolist()

    elif args.model_type == "monot5":
        scorer = MonoT5Scorer(args.model_name, device=device)
        def score_func(q, d): return scorer.score_pairs([q]*len(d), d)

    elif args.model_type == "hf-seqcls":
        scorer = HFSeqClsScorer(args.model_name, device=device, max_length=2048) # Default 2048 for ModernBERT
        def score_func(q, d): return scorer.score_pairs([q]*len(d), d)

    elif args.model_type == "cohere":
        import cohere
        co = cohere.ClientV2()
        print(f"Cohere API: {args.model_name}")
        def score_func(q, d):
            resp = co.rerank(model=args.model_name, query=q, documents=d, top_n=len(d))
            scores = [0.0]*len(d)
            for r in resp.results: scores[r.index] = r.relevance_score
            time.sleep(12.0) # Rate limit safety
            return scores

    elif args.model_type == "voyage":
        import voyageai
        vo = voyageai.Client()
        print(f"Voyage API: {args.model_name}")
        def score_func(q, d):
            resp = vo.rerank(query=q, documents=d, model=args.model_name, top_k=None)
            scores = [0.0]*len(d)
            for r in resp.results: scores[r.index] = r.relevance_score
            time.sleep(10.0)
            return scores

    # 4. Rerank
    eval_for_metrics = []
    
    with open(args.out_tsv, "w", encoding="utf-8") as out_f:
        out_f.write("qid\tmodel\tchunk_id\trank\tscore\trel\n")
        
        for ex in tqdm(cand_examples, desc="Reranking"):
            qid = ex["qid"]
            query = ex["query"]
            candidates = ex["candidates"][:args.max_docs]
            rel_map = {r["chunk_id"]: r.get("rel", 1) for r in ex.get("relevant", [])}
            
            cand_ids = [c["chunk_id"] for c in candidates]
            docs = [id2text.get(cid, "") for cid in cand_ids]
            
            # Skip empty docs if any issues
            valid_indices = [i for i, d in enumerate(docs) if d]
            if not valid_indices:
                continue
                
            valid_docs = [docs[i] for i in valid_indices]
            valid_ids = [cand_ids[i] for i in valid_indices]

            scores = score_func(query, valid_docs)
            
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

    # 5. Metrics
    metrics = compute_metrics_at_ks(eval_for_metrics)
    print("=== Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
