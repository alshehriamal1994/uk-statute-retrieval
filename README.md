# Neural Reranking for UK Statutory Retrieval

This repository contains the official code for the paper **"Neural Reranking for UK Statutory Retrieval"**.
It explains how to reproduce the retrieval, reranking, and evaluation experiments reported in the paper, including the distillation of a **Voyage-law-2** teacher into a **ModernBERT** student model.

## 📂 Data Access

**The dataset is NOT included in this repository.**
Please access the UK Legistlation Corpus and Verified Query Set via the **Durham University Research Data Repository**:
- **DOI**: [10.15128/r14x51hj064](http://doi.org/10.15128/r14x51hj064)

### Expected Data Structure
To run the scripts, place the downloaded data in the `data/` directory or adjust paths accordingly:
- `data/Data.json`: The corpus file.
- `data/queries.jsonl`: The evaluation queries with relevance judgments.

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

**Requirements**: `torch`, `transformers`, `sentence-transformers`, `pyserini`, `scipy`, `pandas`, `faiss-cpu` (or `faiss-gpu`), `matplotlib`.

## 🚀 Usage

All scripts are located in `src/`.

### 1. Retrieval (Candidate Generation)

Generate candidates using BM25 or Dense retrieval (MPNet).

```bash
# BM25
python retrieval/bm25_search.py \
  --data_json data/Data.json \
  --eval_jsonl data/queries.jsonl \
  --output_file outputs/candidates_bm25.jsonl \
  --k 100

# Dense (MPNet)
python retrieval/dense_search.py \
  --data_json data/Data.json \
  --eval_jsonl data/queries.jsonl \
  --output_file outputs/candidates_dense.jsonl \
  --k 100
```

### 2. Reranking

Run inference with various rerankers (CrossEncoders, ColBERT, RankZephyr, or APIs).

**Distilled ModernBERT (Our Student Model):**
```bash
python reranking/rerank.py \
  --model_type hf-seqcls \
  --model_name amal1994/distilled-voyage-modernbert \
  --data_json data/Data.json \
  --candidates outputs/candidates_dense.jsonl \
  --out_tsv results/results_modernbert.tsv
```

**ColBERT:**
```bash
python reranking/rerank_colbert.py \
  --model_name colbert-ir/colbertv2.0 \
  --data_json data/Data.json \
  --candidates outputs/candidates_dense.jsonl \
  --out_tsv results/results_colbert.tsv
```

**RankZephyr (LLM):**
```bash
python reranking/rerank_llm.py \
  --model_name castorini/rank_zephyr_7b_v1_full \
  --data_json data/Data.json \
  --candidates outputs/candidates_dense.jsonl \
  --out_tsv results/results_rankzephyr.tsv
```

### 3. Evaluation & Significance

Compute significance using **Holm-Bonferroni** correction and generate forest plots.

```bash
python evaluation/significance.py \
  --base_file results/results_modernbert.tsv \
  --others results/results_colbert.tsv results/results_rankzephyr.tsv \
  --k 10
```

### 4. Distillation (Training)

Train your own student model using teacher scores (e.g., from Voyage).

```bash
python distillation/train_distill.py \
  --data data/train.jsonl \
  --output_dir checkpoints/distilled_model \
  --model_name nomic-ai/modernbert-embed-base \
  --epochs 3
```





