import math
from typing import List, Dict

def compute_metrics_at_ks(
    examples: List[Dict],
    ks=(5, 10, 25, 50, 100),
) -> Dict[str, float]:
    """
    examples: each has
      - 'relevant': list of {chunk_id, rel}
      - 'reranked': list of (chunk_id, score), sorted desc
    """
    ks = sorted(set(ks))
    n = len(examples)
    sums = {f"Recall@{k}": 0.0 for k in ks}
    sums.update({f"MRR@{k}": 0.0 for k in ks})
    sums.update({f"nDCG@{k}": 0.0 for k in ks})

    for ex in examples:
        rel_map = {r["chunk_id"]: r.get("rel", 1) for r in ex["relevant"]}
        ranked_ids = [cid for cid, _ in ex["reranked"]]
        # CRITICAL FIX: Only count rel > 0 as relevant for Recall/MRR
        relevant_ids = {cid for cid, rel in rel_map.items() if rel > 0}

        # Precompute ideal gains per k for nDCG
        ideal_rels_sorted = sorted(rel_map.values(), reverse=True)
        ideal_prefix_gains = {}
        for k in ks:
            ideal = ideal_rels_sorted[:k]
            if not ideal:
                ideal_prefix_gains[k] = 0.0
            else:
                idcg = 0.0
                for i, rel in enumerate(ideal, start=1):
                    idcg += (2**rel - 1) / math.log2(i + 1)
                ideal_prefix_gains[k] = idcg

        for k in ks:
            top_ids = ranked_ids[:k]

            # Recall@k (graded as "at least one relevant doc in top k" fraction or recall count)
            # Standard definition: |retrieved & relevant| / |relevant|
            if relevant_ids:
                hits = [cid for cid in top_ids if cid in relevant_ids]
                recall = len(hits) / len(relevant_ids)
            else:
                recall = 0.0
            sums[f"Recall@{k}"] += recall

            # MRR@k
            rr = 0.0
            for rank, cid in enumerate(top_ids, start=1):
                if cid in relevant_ids:
                    rr = 1.0 / rank
                    break
            sums[f"MRR@{k}"] += rr

            # nDCG@k
            dcg = 0.0
            for rank, cid in enumerate(top_ids, start=1):
                rel = rel_map.get(cid, 0)
                if rel > 0:
                    dcg += (2**rel - 1) / math.log2(rank + 1)
            idcg = ideal_prefix_gains[k]
            ndcg = dcg / idcg if idcg > 0 else 0.0
            sums[f"nDCG@{k}"] += ndcg

    metrics = {}
    for name, total in sums.items():
        metrics[name] = total / n
    return metrics
