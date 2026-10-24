# %% [markdown]
# # A Ranking SVM Reranker for RAG with Jaddle
# A ranking SVM learns a weight for every BM25 term. This example builds a
# modern retrieval stack around that idea:
#
# 1. **Two first-stage retrievers.** BM25 matches keywords. A small dense
#    embedding model (BAAI's bge-small, 33M parameters, run on the CPU by
#    `fastembed`) matches meaning. Reciprocal rank fusion (RRF) merges their
#    rankings into 100 candidates per query.
# 2. **A reranker** (Joachims, 2002; the LETOR benchmarks). A linear model
#    reorders the candidates using 17 features of each (query, document)
#    pair:
#    - keyword matches: BM25, title BM25, exact-word BM25, TF-IDF
#    - similarities: LSA, and the dense embeddings' cosine
#    - how much of the query a document covers
#    - lengths
#    - the learned term-weight score, fitted below
#
# Unlike per-term weights, these features mean the same thing for every
# query, so what the model learns carries over to queries it has never seen.
#
# The reranker is a ranking SVM again: a QP whose whole regularisation path
# is solved in one batched `jc.make_qp_solver` call. We fit it twice, with
# the keyword features only and with all of them, to measure what the dense
# embeddings add.
#
# It needs the `rag_examples` extra (`pip install "jaddle[rag_examples]"`).
# The embeddings are computed once and cached in `data/`: about 4 minutes for
# SciFact on a CPU. Any BEIR dataset with training labels works via
# `RAG_DATASET`, but larger ones take much longer to embed (FiQA's 57,638
# posts, about 40 minutes).
#
# The data handling, stemmed BM25 and metrics are set up here, so the
# example stands alone.

# %%
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import csv
import io
import json
import time
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import snowballstemmer
from fastembed import TextEmbedding
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import jaddle.jaddle_convex as jc
import jaddle.jaddle_optimisers as jo

jo.configure_jax("float32")

# Any BEIR dataset with training labels works, e.g. "fiqa". Set RAG_DATASET to switch.
DATASET = os.environ.get("RAG_DATASET", "scifact")
DATA_URL = (
    f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{DATASET}.zip"
)
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / f"{DATASET}.zip"

# %% [markdown]
# ## Data, stemmed BM25 and metrics
# BEIR's SciFact claims and abstracts, a fifth of
# the training queries held out for validation, Snowball-stemmed BM25 with
# BEIR's settings, and BEIR's metrics (nDCG@10 and Recall@100).
if not DATA_PATH.exists():
    print(f"Downloading {DATASET} to {DATA_PATH}")
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)

archive = zipfile.ZipFile(DATA_PATH)


def read_jsonl(name):
    lines = archive.read(f"{DATASET}/{name}").decode().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


corpus = read_jsonl("corpus.jsonl")
doc_texts = [f"{d['title']} {d['text']}" for d in corpus]
doc_titles = [d["title"] for d in corpus]  # may be empty (FiQA has none)
doc_index = {d["_id"]: i for i, d in enumerate(corpus)}
query_texts = {q["_id"]: q["text"] for q in read_jsonl("queries.jsonl")}


def read_qrels(split):
    """{query id: set of relevant document indices}."""
    relevant = defaultdict(set)
    rows = csv.reader(
        io.StringIO(archive.read(f"{DATASET}/qrels/{split}.tsv").decode()),
        delimiter="\t",
    )
    next(rows)
    for query_id, doc_id, score in rows:
        if int(score) > 0:
            relevant[query_id].add(doc_index[doc_id])
    return dict(relevant)


train_qrels, test_qrels = read_qrels("train"), read_qrels("test")
if f"{DATASET}/qrels/dev.tsv" in archive.namelist():
    val_qrels = read_qrels("dev")  # the dataset's own validation split
    train_queries, val_queries = sorted(train_qrels), sorted(val_qrels)
else:
    val_qrels = train_qrels
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(sorted(train_qrels))
    val_queries = sorted(shuffled[: len(shuffled) // 5])
    train_queries = sorted(shuffled[len(shuffled) // 5 :])
test_queries = sorted(test_qrels)
print(
    f"{DATASET}: {len(corpus)} documents; {len(train_queries)} train, "
    f"{len(val_queries)} validation and {len(test_queries)} test queries"
)


def batches(queries, size=500):
    for start in range(0, len(queries), size):
        yield queries[start : start + size]


words_of = CountVectorizer(stop_words="english").build_analyzer()
stemmer = snowballstemmer.stemmer("english")
stem_of = {}


def analyze(text):
    """Lowercase, tokenise, drop stop words and stem."""
    stems = []
    for word in words_of(text):
        if word not in stem_of:
            stem_of[word] = stemmer.stemWord(word)
        stems.append(stem_of[word])
    return stems


counter = CountVectorizer(analyzer=analyze)
doc_counts = counter.fit_transform(doc_texts).tocsr().astype(np.float64)
num_docs, num_terms = doc_counts.shape


def bm25_idf(counts):
    doc_freq = np.asarray((counts > 0).sum(axis=0)).ravel()
    return np.log(1 + (counts.shape[0] - doc_freq + 0.5) / (doc_freq + 0.5))


idf = bm25_idf(doc_counts)


def bm25_matrix(counts, idf=idf, k1=0.9, length_norm=0.4):
    """The BM25 weight of every (text, term), as a sparse matrix."""
    length = np.asarray(counts.sum(axis=1)).ravel()
    c = counts.tocoo()
    saturation = (
        c.data
        * (k1 + 1)
        / (
            c.data
            + k1 * (1 - length_norm + length_norm * length[c.row] / length.mean())
        )
    )
    return sp.csr_matrix((saturation * idf[c.col], (c.row, c.col)), shape=counts.shape)


doc_bm25 = bm25_matrix(doc_counts)


def query_terms(queries):
    """Which vocabulary terms each query contains, as a sparse 0/1 matrix."""
    return (
        (counter.transform([query_texts[q] for q in queries]) > 0)
        .astype(np.float64)
        .tocsr()
    )


def bm25_scores(queries, weights=None):
    """BM25 scores of every document for every query; ``weights``
    multiplies each term's BM25 weight."""
    terms = query_terms(queries)
    if weights is not None:
        terms = terms.multiply(weights[None, :]).tocsr()
    return (terms @ doc_bm25.T).toarray()


def ndcg_per_query(scores, queries, qrels, k=10):
    """nDCG@k of each query, from a (queries × documents) score matrix."""
    discount = 1 / np.log2(np.arange(2, k + 2))
    out = []
    for row, q in zip(scores, queries):
        top = np.argpartition(-row, k)[:k]
        top = top[np.argsort(-row[top])]
        relevant = qrels[q]
        out.append(
            np.isin(top, list(relevant))
            @ discount
            / discount[: min(len(relevant), k)].sum()
        )
    return np.array(out)


# %% [markdown]
# ## The learned term-weight score, cross-fitted
# The strongest single feature is the score of the learned term weights:
# BM25 with a multiplier `m_t` on every term, fitted with the
# ranking SVM QP
#
#     minimise ½‖m − 1‖² + (C/T)·Σ ξ_t   subject to   Δ m + ξ ≥ δ,  m ≥ 0,  ξ ≥ 0.
#
# We choose `C` by solving the whole path in one batched `jc.make_qp_solver`
# call, and keeping the value with the best validation nDCG@10.
#
# Using the score as a reranker feature needs care. Weights fitted on all the
# training queries score those same queries' relevant documents too well, and
# the reranker would learn to trust the feature more than it deserves. So we
# **cross-fit**: the training queries are split into four folds, and each
# fold's feature comes from weights fitted on the other three, with
# `jc.quadratic_program`. Validation and test queries use weights fitted on
# all the training queries. We also record what fraction of each query's
# terms got a learned weight. That tells the reranker how much the learned
# score knows about the query.
TERM_MARGIN, TERM_NEGATIVES = 0.5, 20
TERM_C_VALUES = np.logspace(2, 5, 13)


def term_qp(queries):
    """The term-weight QP's matrices for ``queries``, its linear cost as a
    function of C, and the learned terms."""
    positive_scores, rows = [], []
    for batch in batches(queries):
        for row, terms, q in zip(bm25_scores(batch), query_terms(batch), batch):
            relevant = train_qrels[q]
            k = TERM_NEGATIVES + len(relevant)
            top = np.argpartition(-row, k)[:k]
            negatives = [d for d in top[np.argsort(-row[top])] if d not in relevant][
                :TERM_NEGATIVES
            ]
            positive_scores += [row[d] for d in relevant]
            for p in relevant:
                positive = terms.multiply(doc_bm25[p])
                rows += [positive - terms.multiply(doc_bm25[n]) for n in negatives]
    Delta = sp.vstack(rows).tocsr() / np.median(positive_scores)
    learned = np.unique(Delta.indices)
    Delta = Delta[:, learned]
    T, p = Delta.shape
    matrices = dict(
        Q=sp.diags(np.r_[np.ones(p), np.zeros(T)]).tocsr(),
        A_ineq=sp.hstack([-Delta, -sp.eye(T)]).tocsr(),
        lower_bounds=np.zeros(p + T),
        upper_bounds=np.full(p + T, np.inf),
    )
    return matrices, lambda C: np.r_[-np.ones(p), C / T * np.ones(T)], learned


def as_model(m, learned):
    """Full-vocabulary multipliers, and a 0/1 mask of the learned terms."""
    weights, seen = np.ones(num_terms), np.zeros(num_terms)
    weights[learned] = np.maximum(np.asarray(m, dtype=np.float64)[: len(learned)], 0)
    seen[learned] = 1
    return weights, seen


start = time.perf_counter()
matrices, cost, learned = term_qp(train_queries)
num_term_triplets = matrices["A_ineq"].shape[0]
term_solver = jc.make_qp_solver(**matrices, max_epochs=1000)
solve_term_path = jax.jit(
    jax.vmap(lambda c, b: term_solver(c, b_ineq=b), in_axes=(0, None))
)
term_path = solve_term_path(
    jnp.asarray(np.stack([cost(C) for C in TERM_C_VALUES])),
    jnp.full(num_term_triplets, -TERM_MARGIN),
)
term_models = [as_model(x, learned) for x in term_path.solution.primal]
term_val = [
    ndcg_per_query(bm25_scores(val_queries, w), val_queries, val_qrels).mean()
    for w, _ in term_models
]
best_term = int(np.argmax(term_val))
term_C, full_model = TERM_C_VALUES[best_term], term_models[best_term]
print(
    f"Term-weight path: {len(TERM_C_VALUES)} QPs with {num_term_triplets} triplets, "
    f"converged {int(term_path.converged.sum())}/{len(TERM_C_VALUES)}. Best C {term_C:.3g}: "
    f"validation nDCG@10 {term_val[best_term]:.3f} "
    f"(BM25 {ndcg_per_query(bm25_scores(val_queries), val_queries, val_qrels).mean():.3f})"
)

folds = np.array_split(np.random.default_rng(1).permutation(len(train_queries)), 4)
term_model = {}
for fold in folds:
    in_fold = set(fold)
    matrices, cost, learned = term_qp(
        [q for i, q in enumerate(train_queries) if i not in in_fold]
    )
    T = matrices["A_ineq"].shape[0]
    result = jc.quadratic_program(
        c=cost(term_C), b_ineq=np.full(T, -TERM_MARGIN), max_epochs=1000, **matrices
    )
    assert result["converged"]
    model = as_model(result["solution"].primal, learned)
    term_model.update({train_queries[i]: model for i in fold})
term_model.update({q: full_model for q in val_queries + test_queries})
print(f"Fitted the term-weight models in {time.perf_counter() - start:.1f} s")

# %% [markdown]
# ## Dense embeddings and hybrid candidates
# bge-small maps each text to a 384-dimensional unit vector, with queries and
# documents embedded slightly differently (`query_embed` adds bge's search
# instruction). A document's dense score is the cosine between the two
# vectors.
#
# Reciprocal rank fusion scores each document `1/(60 + BM25 rank) + 1/(60 +
# dense rank)`. A document both retrievers rank well rises to the top, and
# one either retriever finds is likely to make the top 100. Unlike adding
# the raw scores, it doesn't care that the two scores are on different scales.
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
DENSE_CACHE = DATA_PATH.parent / f"{DATASET}_bge-small-en-v1.5.npz"


def embed_corpus():
    """Embed every document and query with bge-small and cache the vectors."""
    model = TextEmbedding(DENSE_MODEL)
    texts = [text.strip() for text in doc_texts]
    order = np.argsort(
        [len(text) for text in texts]
    )  # similar lengths batch with less padding
    doc_emb = np.zeros((len(texts), 384), np.float32)
    start = time.perf_counter()
    done = 0
    for block in np.array_split(order, max(1, len(order) // 1024)):
        doc_emb[block] = np.array(
            list(model.passage_embed([texts[i] for i in block], batch_size=32))
        )
        done += len(block)
        print(
            f"Embedded {done}/{len(texts)} documents ({done / (time.perf_counter() - start):.0f}/s)"
        )
    query_ids = sorted(query_texts)
    query_emb = np.array(
        list(model.query_embed([query_texts[q] for q in query_ids])), np.float32
    )
    np.savez(
        DENSE_CACHE,
        doc_ids=np.array([d["_id"] for d in corpus]),
        doc_emb=doc_emb,
        query_ids=np.array(query_ids),
        query_emb=query_emb,
    )


if not DENSE_CACHE.exists():
    print(
        f"Embedding {DATASET} with {DENSE_MODEL}; the vectors are cached in {DENSE_CACHE}"
    )
    embed_corpus()
dense = np.load(DENSE_CACHE)
assert list(dense["doc_ids"]) == [d["_id"] for d in corpus]
doc_dense = dense["doc_emb"]
query_dense = dict(zip(dense["query_ids"], dense["query_emb"]))


def dense_scores(queries):
    """Cosine of every document's embedding with each query's."""
    return np.stack([query_dense[q] for q in queries]) @ doc_dense.T


def rrf_scores(bm25, dense, k=60):
    """Reciprocal rank fusion of two (queries × documents) score matrices."""
    bm25_rank = np.argsort(np.argsort(-bm25, axis=1), axis=1)
    dense_rank = np.argsort(np.argsort(-dense, axis=1), axis=1)
    return 1 / (k + bm25_rank) + 1 / (k + dense_rank)


def full_corpus_metrics(score_fn, queries, qrels):
    """Mean nDCG@10 and Recall@100 of ranking the whole corpus by ``score_fn``."""
    ndcg, recall = [], []
    for batch in batches(queries):
        scores = score_fn(batch)
        ndcg.append(ndcg_per_query(scores, batch, qrels))
        for row, q in zip(scores, batch):
            recall.append(
                np.isin(list(qrels[q]), np.argpartition(-row, 100)[:100]).mean()
            )
    return np.concatenate(ndcg).mean(), np.mean(recall)


print(f"{'Full-corpus retrieval':24s} {'test nDCG@10':>12s} {'Recall@100':>11s}")
for name, score_fn in [
    ("BM25", bm25_scores),
    ("Dense (bge-small)", dense_scores),
    ("RRF of BM25 and dense", lambda qs: rrf_scores(bm25_scores(qs), dense_scores(qs))),
]:
    ndcg, recall = full_corpus_metrics(score_fn, test_queries, test_qrels)
    print(f"{name:24s} {ndcg:12.3f} {recall:11.3f}")

# %% [markdown]
# ## Candidates and features
# Each query's candidates are RRF's top 100. Reranking can only reorder them,
# so their Recall@100 caps what any reranker can find: 0.97 on SciFact,
# against 0.92 for BM25's top 100. Each (query, candidate) pair gets 17
# features:
#
# | Feature | What it measures |
# |---|---|
# | BM25, and BM25 / the query's top score | keyword match, absolute and relative |
# | title BM25 | keyword match on the title alone |
# | exact BM25 | BM25 without stemming: exact word forms |
# | TF-IDF cosine | another weighting of the same keywords |
# | LSA cosine | similarity in a 256-dimensional SVD of TF-IDF, which can match related words that never co-occur in the pair |
# | term cover, idf cover | fraction of the query's terms in the document, unweighted and idf-weighted |
# | bigram match | how many of the query's consecutive stem pairs appear in the document |
# | log document length, log query length | context for the other features |
# | learned BM25, and learned / top | the cross-fitted learned term-weight score |
# | seen fraction | fraction of the query's terms with a learned weight |
# | dense cosine, and dense − the query's top | the embeddings' similarity, absolute and relative |
# | RRF | the fused rank score |
#
# The first 14 are the keyword features; the last 3 need the embeddings.
#
# Features are standardised with the training pairs' means and standard
# deviations.
NUM_CANDIDATES = 100
FEATURE_NAMES = [
    "BM25", "BM25 / top", "title BM25", "exact BM25", "TF-IDF cosine",
    "LSA cosine", "term cover", "idf cover", "bigram match", "log doc length",
    "log query length", "learned BM25", "learned / top", "seen fraction",
    "dense cosine", "dense - top", "RRF",
]  # fmt: skip
KEYWORD_FEATURES = list(range(14))

tfidf = TfidfTransformer(sublinear_tf=True).fit(doc_counts)
doc_tfidf = tfidf.transform(doc_counts)
svd = TruncatedSVD(256, random_state=0).fit(doc_tfidf)


def unit_rows(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


doc_lsa = unit_rows(svd.transform(doc_tfidf))
title_bm25 = bm25_matrix(counter.transform(doc_titles).astype(np.float64))
doc_has_term = (doc_counts > 0).astype(np.float64).T.tocsr()
log_doc_length = np.log1p(np.asarray(doc_counts.sum(axis=1)).ravel())

exact_counter = CountVectorizer(stop_words="english")
exact_counts = exact_counter.fit_transform(doc_texts).astype(np.float64)
exact_bm25 = bm25_matrix(exact_counts, idf=bm25_idf(exact_counts))


def bigrams(text):
    stems = analyze(text)
    return [f"{a} {b}" for a, b in zip(stems, stems[1:])]


bigram_counter = CountVectorizer(analyzer=bigrams)
doc_has_bigram = (
    (bigram_counter.fit_transform(doc_texts) > 0).astype(np.float64).T.tocsr()
)


def candidates_and_features(queries):
    """Candidates and features for ``queries``, computed in batches."""
    parts = [features_batch(batch) for batch in batches(queries, 250)]
    return np.concatenate([c for c, _ in parts]), np.concatenate([x for _, x in parts])


def features_batch(queries):
    texts = [query_texts[q] for q in queries]
    terms = query_terms(queries)
    bm25 = (terms @ doc_bm25.T).toarray()
    dense = dense_scores(queries)
    rrf = rrf_scores(bm25, dense)
    candidates = np.argsort(-rrf, axis=1)[:, :NUM_CANDIDATES]

    def pick(matrix):
        matrix = matrix.toarray() if sp.issparse(matrix) else matrix
        return np.take_along_axis(matrix, candidates, axis=1)

    query_tfidf = tfidf.transform(counter.transform(texts))
    # max(·, 1) and the guards below: a query can match no indexed term.
    num_query_terms = np.maximum(np.asarray(terms.sum(axis=1)).ravel(), 1)
    query_idf = terms.multiply(idf[None, :]).tocsr()
    learned = np.zeros_like(candidates, dtype=np.float64)
    seen = np.zeros(len(queries))
    for i, q in enumerate(queries):
        weights, mask = term_model[q]
        learned[i] = (
            terms[i].multiply(weights[None, :]) @ doc_bm25[candidates[i]].T
        ).toarray()
        seen[i] = terms[i].multiply(mask[None, :]).sum() / num_query_terms[i]
    exact_terms = (exact_counter.transform(texts) > 0).astype(np.float64)
    query_bigrams = (bigram_counter.transform(texts) > 0).astype(np.float64)
    top = pick(bm25)
    features = [
        top,
        top / np.maximum(top.max(axis=1, keepdims=True), 1e-12),
        pick(terms @ title_bm25.T),
        pick(exact_terms @ exact_bm25.T),
        pick(query_tfidf @ doc_tfidf.T),
        pick(unit_rows(svd.transform(query_tfidf)) @ doc_lsa.T),
        pick(terms @ doc_has_term) / num_query_terms[:, None],
        pick(query_idf @ doc_has_term)
        / np.maximum(np.asarray(query_idf.sum(axis=1)), 1e-12),
        pick(query_bigrams @ doc_has_bigram),
        log_doc_length[candidates],
        np.repeat(np.log(num_query_terms)[:, None], NUM_CANDIDATES, axis=1),
        learned,
        learned / np.maximum(learned.max(axis=1, keepdims=True), 1e-12),
        np.repeat(seen[:, None], NUM_CANDIDATES, axis=1),
        pick(dense),
        pick(dense) - pick(dense).max(axis=1, keepdims=True),
        pick(rrf),
    ]
    return candidates, np.stack(features, axis=-1)


train_cand, X_train = candidates_and_features(train_queries)
val_cand, X_val = candidates_and_features(val_queries)
test_cand, X_test = candidates_and_features(test_queries)
mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
std[std == 0] = 1  # a constant feature, e.g. title BM25 when there are no titles
X_train, X_val, X_test = [(X - mean) / std for X in (X_train, X_val, X_test)]
assert all(np.isfinite(X).all() for X in (X_train, X_val, X_test))


def rerank_ndcg(scores, candidates, queries, qrels):
    """Per-query nDCG@10 of reordering ``candidates`` by ``scores``."""
    full = np.full((len(queries), num_docs), -np.inf)
    np.put_along_axis(full, candidates, scores, axis=1)
    return ndcg_per_query(full, queries, qrels)


ceiling = np.mean(
    [np.isin(list(test_qrels[q]), c).mean() for q, c in zip(test_queries, test_cand)]
)
print(f"Test Recall@100 of the candidates (the reranking ceiling): {ceiling:.3f}")
print(f"{'Feature alone':18s} {'val nDCG@10':>12s} {'test nDCG@10':>13s}")
for j, name in enumerate(FEATURE_NAMES):
    val = rerank_ndcg(X_val[..., j], val_cand, val_queries, val_qrels).mean()
    test = rerank_ndcg(X_test[..., j], test_cand, test_queries, test_qrels).mean()
    print(f"{name:18s} {val:12.3f} {test:13.3f}")

# %% [markdown]
# ## Training triplets
# For each training query, every relevant candidate is paired with 5
# non-relevant candidates drawn at random. The negatives are random rather
# than the hardest ones, because a reranker sees the whole top 100, not just
# its hardest part. That gives about 3,600 triplets `(x⁺ₜ, x⁻ₜ)` on SciFact.
NUM_NEGATIVES = 5
sampler = np.random.default_rng(1)
X_pos, X_neg = [], []
for i, q in enumerate(train_queries):
    is_relevant = np.isin(train_cand[i], list(train_qrels[q]))
    negatives = np.flatnonzero(~is_relevant)
    for p in np.flatnonzero(is_relevant):
        for n in sampler.choice(negatives, NUM_NEGATIVES, replace=False):
            X_pos.append(X_train[i, p])
            X_neg.append(X_train[i, n])
X_pos, X_neg = np.array(X_pos), np.array(X_neg)
D = X_pos - X_neg
num_triplets, num_features = D.shape
num_keyword = len(KEYWORD_FEATURES)
print(f"{num_triplets} training triplets")

# %% [markdown]
# ## The reranker's QP
# The reranker scores `f(x) = vᵀx`. The ranking SVM asks every relevant
# candidate to score at least 1 above its paired negative, with a slack
# `ξ_t` for each triplet that falls short:
#
#     minimise    ½‖v‖² + (C/T)·Σ ξ_t
#     subject to  (x⁺ₜ − x⁻ₜ)ᵀv + ξ_t ≥ 1,   ξ ≥ 0.
#
# This is the term-weight QP above with one free weight per feature in
# place of one per term, and no pull towards BM25. In the form `½ xᵀQx + cᵀx`:
# - `x = (v, ξ)`
# - `Q = diag(1, …, 1, 0, …, 0)`
# - `c = (0, …, 0, C/T, …, C/T)`
# - the constraint is `[−D, −I] x ≤ −1`, where `D`'s rows are `x⁺ₜ − x⁻ₜ`
#
# `C` appears only in `c`, so one `vmap`-ed `jc.make_qp_solver` call solves
# the whole path. We fit two rerankers this way: one on the 14 keyword
# features, and one on all 17.
C_VALUES = np.logspace(-2, 3, 11)


def fit_reranker(columns):
    """Solve the ranking SVM path on the feature ``columns`` and choose C by
    validation nDCG@10. Returns the chosen weights (zero off ``columns``),
    the validation path and the batched result."""
    Dc = D[:, columns]
    T, F = Dc.shape
    solver = jc.make_qp_solver(
        sp.diags(np.r_[np.ones(F), np.zeros(T)]).tocsr(),
        A_ineq=sp.hstack([-sp.csr_matrix(Dc), -sp.eye(T)]).tocsr(),
        lower_bounds=np.r_[np.full(F, -np.inf), np.zeros(T)],
        upper_bounds=np.full(F + T, np.inf),
        max_epochs=2000,
    )
    solve_path = jax.jit(jax.vmap(lambda c, b: solver(c, b_ineq=b), in_axes=(0, None)))
    c_batch = jnp.asarray(
        np.stack([np.r_[np.zeros(F), C / T * np.ones(T)] for C in C_VALUES])
    )
    b_ineq = jnp.full(T, -1.0)
    jax.block_until_ready(solve_path(c_batch, b_ineq))  # compile
    start = time.perf_counter()
    path = jax.block_until_ready(solve_path(c_batch, b_ineq))
    print(
        f"{F} features: {len(C_VALUES)} QPs ({F + T} variables and {T} constraints each) "
        f"solved in {time.perf_counter() - start:.2f} s on {jax.devices()[0].platform}; "
        f"converged {int(path.converged.sum())}/{len(C_VALUES)}"
    )
    weights = np.asarray(path.solution.primal, dtype=np.float64)[:, :F]
    val = np.array(
        [
            rerank_ndcg(
                X_val[..., columns] @ w, val_cand, val_queries, val_qrels
            ).mean()
            for w in weights
        ]
    )
    best = int(np.argmax(val))
    v = np.zeros(num_features)
    v[columns] = weights[best]
    print(f"   best C {C_VALUES[best]:.3g}: validation nDCG@10 {val[best]:.3f}")
    return dict(
        v=v, C=C_VALUES[best], best=best, val=val, path=path, D=Dc, columns=columns
    )


keyword = fit_reranker(KEYWORD_FEATURES)
hybrid = fit_reranker(list(range(num_features)))
v, best_C = hybrid["v"], hybrid["C"]
val_of = lambda name: rerank_ndcg(
    X_val[..., FEATURE_NAMES.index(name)], val_cand, val_queries, val_qrels
).mean()

# %% [markdown]
# ### Checking the solve, and reading the duals
# If CVXPY is installed, we solve the chosen problem again with Clarabel and
# compare optimal values.
#
# The margin constraints' duals lie in `[0, C/T]`:
# - **0:** the triplet is ranked correctly with room to spare, and doesn't
#   shape `v`.
# - **Strictly between 0 and `C/T`:** the triplet sits exactly on the margin.
# - **At `C/T`:** the triplet falls short of the margin.
# We check the reranker with all 17 features.
hinge_weight = best_C / num_triplets
xi = np.maximum(1 - D @ v, 0)
jaddle_objective = 0.5 * v @ v + hinge_weight * xi.sum()
try:
    import cvxpy as cp

    v_var, xi_var = cp.Variable(num_features), cp.Variable(num_triplets, nonneg=True)
    reference = cp.Problem(
        cp.Minimize(0.5 * cp.sum_squares(v_var) + hinge_weight * cp.sum(xi_var)),
        [D @ v_var + xi_var >= 1],
    )
    reference.solve(solver=cp.CLARABEL)
    gap = abs(jaddle_objective - reference.value) / abs(reference.value)
    print(
        f"Optimal value: Jaddle {jaddle_objective:.6g}, Clarabel {reference.value:.6g} "
        f"(relative gap {gap:.1e})"
    )
    assert gap < 1e-3
except ImportError:
    print(
        f"Optimal value: Jaddle {jaddle_objective:.6g} (install cvxpy to check it against Clarabel)"
    )

duals = np.asarray(hybrid["path"].solution.dual_ineq[hybrid["best"]], dtype=np.float64)
tol = 1e-3 * hinge_weight
print(
    f"Triplets: {np.sum(duals <= tol)} beyond the margin, "
    f"{np.sum((duals > tol) & (duals < hinge_weight - tol))} on it, "
    f"{np.sum(duals >= hinge_weight - tol)} short of it"
)


# %% [markdown]
# ## Results
# Test nDCG@10 for single signals and the two rerankers, all ranking the
# same RRF candidates. The hybrid reranker's gains have 95% intervals from a
# paired bootstrap over the test queries.
def single(name):
    return rerank_ndcg(
        X_test[..., FEATURE_NAMES.index(name)], test_cand, test_queries, test_qrels
    )


test_rows = {
    "BM25": single("BM25"),
    "Learned term weights": single("learned BM25"),
    "Dense": single("dense cosine"),
    "RRF": single("RRF"),
    "Keyword reranker": rerank_ndcg(
        X_test @ keyword["v"], test_cand, test_queries, test_qrels
    ),
    "Hybrid reranker": rerank_ndcg(X_test @ v, test_cand, test_queries, test_qrels),
}


def bootstrap_interval(diff, draws=2000):
    boot = np.random.default_rng(0)
    means = [diff[boot.integers(0, len(diff), len(diff))].mean() for _ in range(draws)]
    return np.percentile(means, [2.5, 97.5])


print(f"{'':22s} {'test nDCG@10':>12s}")
for name, scores in test_rows.items():
    print(f"{name:22s} {scores.mean():12.3f}")
for baseline in ("RRF", "Keyword reranker", "Learned term weights"):
    diff = test_rows["Hybrid reranker"] - test_rows[baseline]
    low, high = bootstrap_interval(diff)
    print(
        f"Hybrid reranker gain over {baseline}: {diff.mean():+.3f} "
        f"(95% interval [{low:+.3f}, {high:+.3f}])"
    )

# %%
fig, (ax_path, ax_weights) = plt.subplots(1, 2, figsize=(13, 4.5))
ax_path.semilogx(
    C_VALUES, hybrid["val"], "o-", color="C0", label="Hybrid reranker (17 features)"
)
ax_path.semilogx(
    C_VALUES, keyword["val"], "s-", color="C1", label="Keyword reranker (14 features)"
)
ax_path.axhline(val_of("RRF"), color="C7", linestyle="--", label="RRF")
ax_path.axhline(val_of("BM25"), color="C7", linestyle=":", label="BM25")
ax_path.set_xlabel("C (weight on the ranking loss)")
ax_path.set_ylabel("Validation nDCG@10")
ax_path.set_title("Regularisation paths, one batched call each")
ax_path.legend()

order = np.argsort(v)
ax_weights.barh(
    np.array(FEATURE_NAMES)[order], v[order], color=np.where(v[order] >= 0, "C0", "C3")
)
ax_weights.axvline(0, color="black", linewidth=0.8)
ax_weights.set_xlabel("Weight (features standardised)")
ax_weights.set_title("What the hybrid reranker learned")
plt.tight_layout()
plt.show()

# %% [markdown]
# Results from a run on a laptop GPU, on SciFact (test nDCG@10):
#
# | | nDCG@10 |
# |---|---|
# | BM25 | 0.681 |
# | Learned term weights | 0.708 |
# | Dense (bge-small) | 0.720 |
# | RRF of BM25 and dense | 0.731 |
# | Keyword reranker (14 features) | 0.724 |
# | **Hybrid reranker (17 features)** | **0.770** |
#
# - **The hybrid reranker is clearly best.** It gains 4.0 points over RRF
#   (95% interval [+2.1, +6.0]) and 4.6 over the keyword reranker
#   ([+2.3, +7.1]), and validation agrees (0.817 against RRF's 0.791).
# - **The dense embeddings do two jobs.**
#   - They find candidates BM25 misses: RRF's top 100 holds 96.5% of the
#     relevant abstracts, against 92% for BM25's.
#   - They give the reranker a signal the keyword features lack. The keyword
#     reranker can't beat RRF; its best validation score is below RRF's.
# - **Neither signal is enough alone.** Dense retrieval (0.720) and the
#   learned term weights (0.708) are close. The reranker combines them, and
#   that combination is what lifts it to 0.770.
#
# **What it learned** (right-hand plot). The dense similarity carries the
# most weight, absolute and relative to the query's best candidate. The
# learned term score and idf-weighted coverage come next: they add the
# precision of exact keyword matches. LSA gets no weight, because the dense
# embeddings supersede it. Some keyword features get negative weights, but
# the keyword features are strongly correlated, so a negative weight here
# corrects for overlap with the others rather than meaning that matches hurt.
# Query length and seen fraction get none, as before, because they are the
# same for every candidate of a query.
#
# **For scale**, 0.770 is in the range published for embedding models
# built on 7B language models (about 0.77–0.80 in published work), from
# a 33M-parameter embedding model, BM25 and a linear model trained by a
# convex QP. The caveat applies: the reranker is
# trained on SciFact's labels, while most published models are zero-shot.
#
# **Cost.** Each reranker's path is 11 QPs with about 3,600 variables, solved
# in about 3 seconds on the GPU and matching Clarabel to about 1e-5. The run
# takes about 35 seconds once the embeddings are cached.

# %%
s
