import argparse
import json
import re
import torch
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

from evaluation.metrics import compute_metrics_at_ks

class RankZephyrReranker:
    def __init__(self, model_name: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading RankZephyr model {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        self.model.eval()

    def _build_prompt(self, query: str, docs: List[str]) -> str:
        truncated_docs = []
        for d in docs:
            truncated_docs.append(d[:600] if len(d) > 600 else d)
            
        doc_block = "\n".join([f"[{i}] {d}" for i, d in enumerate(truncated_docs, start=1)])
        
        prompt = (
            "You are a legal information retrieval assistant. "
            "Your task is to rerank candidate law passages for a user query by relevance.\n\n"
            f"Query:\n{query}\n\n"
            "Candidate passages:\n"
            f"{doc_block}\n\n"
            "Please sort these passages from most relevant to least relevant to the query.\n"
            "Return ONLY a JSON array of the indices (1-based) in the new order, like:\n"
            "[3, 1, 2]\n\n"
            "Answer:\n"
        )
        return prompt

    def _parse_order(self, text: str, num_docs: int) -> List[int]:
        m = re.search(r'\[.*?\]', text, re.DOTALL)
        if m:
            try:
                indices = [int(x) for x in json.loads(m.group(0))]
            except:
                indices = []
        else:
            indices = [int(x) for x in re.findall(r'\d+', text)]
            
        seen = set()
        cleaned = []
        for i in indices:
            if 1 <= i <= num_docs and i not in seen:
                seen.add(i)
                cleaned.append(i)
        
        remaining = [i for i in range(1, num_docs + 1) if i not in seen]
        return cleaned + remaining

    def rerank(self, query: str, docs: List[str]) -> List[int]:
        prompt = self._build_prompt(query, docs)
        # Scientific Note: Mistral-7B (base of RankZephyr) has 8k context. 
        # We increase max_length to 8192 to accommodate more docs (approx 50-60). 
        # For full 100 docs, a sliding window approach is recommended but this captures more than 4096.
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=128, do_sample=False, temperature=0.0,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated[len(prompt):]
        return self._parse_order(answer, len(docs))

def main():
    parser = argparse.ArgumentParser(description="Run RankZephyr (LLM) Reranker")
    parser.add_argument("--candidates", type=str, required=True)
    parser.add_argument("--data_json", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="castorini/rank_zephyr_7b_v1_full")
    parser.add_argument("--out_tsv", type=str, default="results_rankzephyr.tsv")
    parser.add_argument("--max_docs", type=int, default=100)
    args = parser.parse_args()

    set_seed(42)

    data_path = Path(args.data_json)
    cand_path = Path(args.candidates)
    
    Path(args.out_tsv).parent.mkdir(parents=True, exist_ok=True)

    with data_path.open("r", encoding="utf-8") as f:
        corpus = json.load(f)
    id2text = {item.get("metadata", {}).get("chunk_id", item.get("metadata", {}).get("doc_id")): item.get("content", "") for item in corpus}

    cand_examples = []
    with cand_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): cand_examples.append(json.loads(line))

    reranker = RankZephyrReranker(args.model_name)
    eval_for_metrics = []

    with open(args.out_tsv, "w", encoding="utf-8") as out_f:
        out_f.write("qid\tmodel\tchunk_id\trank\tscore\trel\n")
        
        for ex in tqdm(cand_examples, desc="RankZephyr Reranking"):
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

            order = reranker.rerank(query, valid_docs)
            ordered_ids = [valid_ids[i-1] for i in order]
            
            scored = []
            for rank_idx, cid in enumerate(ordered_ids, start=1):
                score = float(len(ordered_ids) - rank_idx + 1)
                scored.append((cid, score))
            
            eval_for_metrics.append({
                "qid": qid,
                "relevant": ex.get("relevant", []),
                "reranked": scored
            })
            
            for rank, (cid, s) in enumerate(scored, start=1):
                rel = rel_map.get(cid, 0)
                out_f.write(f"{qid}\t{args.model_name}\t{cid}\t{rank}\t{s}\t{rel}\n")

    metrics = compute_metrics_at_ks(eval_for_metrics)
    print("=== RankZephyr Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
