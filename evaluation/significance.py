import pandas as pd
import numpy as np
import argparse
from scipy.stats import wilcoxon
from pathlib import Path

# ============ FUNCTIONS ============
def ndcg_at_k(rels, k=10):
    rels = np.asarray(rels, dtype=float)
    if len(rels) == 0:
        return 0.0
    k = min(k, len(rels))
    rels_k = rels[:k]
    gains = (2.0 ** rels_k - 1.0) / np.log2(np.arange(2, k + 2))
    dcg = gains.sum()
    sorted_rels = np.sort(rels)[::-1]
    ideal_k = min(k, len(sorted_rels))
    ideal_rels_k = sorted_rels[:ideal_k]
    ideal_gains = (2.0 ** ideal_rels_k - 1.0) / np.log2(np.arange(2, ideal_k + 2))
    idcg = ideal_gains.sum()
    return 0.0 if idcg == 0 else dcg / idcg

def load_ndcg_per_query(path, k=10):
    df = pd.read_csv(path, sep="\t")
    # ensure numeric rank
    df["rank"] = pd.to_numeric(df["rank"], errors='coerce').fillna(0).astype(int)
    groups = df.groupby("qid")
    qids = sorted(groups.groups.keys())
    scores = []
    for qid in qids:
        g = groups.get_group(qid).sort_values("rank")
        rels = g["rel"].to_numpy()
        scores.append(ndcg_at_k(rels, k=k))
    return np.array(scores), qids

def cohen_d(x, y):
    diff = x - y
    return diff.mean() / diff.std(ddof=1)

def win_tie_loss(x, y, eps=1e-9):
    wins = np.sum(x > y + eps)
    losses = np.sum(x + eps < y)
    ties = np.sum(np.abs(x - y) <= eps)
    return wins, ties, losses

def bootstrap_ci(diff, n_boot=10000, alpha=0.05):
    rng = np.random.default_rng(0)
    n = len(diff)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        means.append(diff[idx].mean())
    return np.percentile(means, alpha/2*100), np.percentile(means, (1-alpha/2)*100)

def main():
    parser = argparse.ArgumentParser(description="Statistical Significance Tests (Holm-Bonferroni)")
    parser.add_argument("--base_file", type=str, required=True, help="Baseline model TSV (e.g. Distilled Voyage)")
    parser.add_argument("--others", nargs="+", required=True, help="List of other model TSVs to compare against")
    parser.add_argument("--k", type=int, default=10, help="nDCG@K cutoff")
    args = parser.parse_args()

    print(f"Base model: {Path(args.base_file).name}")
    base_scores, base_qids = load_ndcg_per_query(args.base_file, k=args.k)
    print(f"Number of queries: {len(base_scores)}")
    
    results = []
    
    for path in args.others:
        p_obj = Path(path)
        print(f"\nProcessing: {p_obj.name}")
        try:
            scores, qids = load_ndcg_per_query(path, k=args.k)
            if qids != base_qids:
                print(f"  WARNING: Query ID mismatch or count mismatch ({len(qids)} vs {len(base_qids)})")
                # Attempt to align by intersection if needed, but for now mostly strict
                if len(qids) != len(base_qids):
                    print("Skipping due to length mismatch")
                    continue
            
            # Wilcoxon
            stat, p = wilcoxon(base_scores, scores, zero_method="wilcox")
            d = cohen_d(base_scores, scores)
            wins, ties, losses = win_tie_loss(base_scores, scores)
            diff = base_scores - scores
            ci_low, ci_high = bootstrap_ci(diff)
            
            results.append({
                "model": p_obj.stem.replace("results_", "").replace("dense_", "").replace("bm25_", ""),
                "p": p,
                "cohen_d": d,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "mean_diff": diff.mean(),
                "ci_low": ci_low,
                "ci_high": ci_high,
            })
            
        except Exception as e:
            print(f"  ERROR processing {path}: {e}")

    # Holm-Bonferroni correction
    if not results:
        print("No results to analyze.")
        return

    m = len(results)
    idx_sorted = sorted(range(m), key=lambda i: results[i]["p"])
    for rank, idx in enumerate(idx_sorted, 1):
        raw = results[idx]["p"]
        results[idx]["p_holm"] = min(1.0, raw * (m - rank + 1))

    print(f"\n{'='*80}")
    print(f"SIGNIFICANCE TEST RESULTS (nDCG@{args.k})")
    print(f"{'='*80}\n")
    
    for r in sorted(results, key=lambda x: x["p"]):
        sig_mark = "✅ p < 0.05" if r['p_holm'] < 0.05 else "❌ NOT SIG"
        print(f"Model: {r['model']}")
        print(f"  Raw p-value          : {r['p']:.4g}")
        print(f"  Holm-Bonferroni p    : {r['p_holm']:.4g}  {sig_mark}")
        print(f"  Cohen's d            : {r['cohen_d']:.3f}")
        print(f"  Mean diff            : {r['mean_diff']:.4f}")
        print(f"  95% bootstrap CI     : [{r['ci_low']:.4f}, {r['ci_high']:.4f}]")
        print(f"  Win / Tie / Loss     : {r['wins']} / {r['ties']} / {r['losses']}")
        print()

if __name__ == "__main__":
    main()
