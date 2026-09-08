from __future__ import annotations

import argparse
from collections import defaultdict
import logging
import random
from typing import Any

from datasets import Audio, Dataset, load_dataset
from huggingface_hub import HfApi, whoami

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SOURCE_DATASET = "mteb/AVE-Dataset"
SOURCE_REVISION = "f6eb93b4e89456277a242583b5565b801bc1981d"
A2V_REPO = "iamfortytwo/AVE-A2V"
V2A_REPO = "iamfortytwo/AVE-V2A"


def partition_ave_dataset(
    ds: Dataset, queries_per_class: int = 4, seed: int = 42
) -> tuple[list[int], list[int], list[str], list[str], list[dict[str, Any]]]:
    """Group AVE-Dataset clips by label and partition into disjoint queries and corpus."""
    label_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(ds["label"]):
        label_to_indices[label].append(idx)

    rng = random.Random(seed)
    query_indices, corpus_indices = [], []
    query_ids, corpus_ids = [], []
    query_to_label, corpus_to_label = {}, {}

    for label in sorted(label_to_indices.keys()):
        indices = list(label_to_indices[label])
        rng.shuffle(indices)
        q_idx = indices[:queries_per_class]
        c_idx = indices[queries_per_class:]
        for i, idx in enumerate(q_idx):
            qid = f"q-{label}-{i}"
            query_indices.append(idx)
            query_ids.append(qid)
            query_to_label[qid] = label
        for j, idx in enumerate(c_idx):
            cid = f"c-{label}-{j}"
            corpus_indices.append(idx)
            corpus_ids.append(cid)
            corpus_to_label[cid] = label

    qrels_rows = []
    for qid, q_label in query_to_label.items():
        for cid, c_label in corpus_to_label.items():
            if q_label == c_label:
                qrels_rows.append({"query-id": qid, "corpus-id": cid, "score": 1})

    return query_indices, corpus_indices, query_ids, corpus_ids, qrels_rows


def create_and_push_datasets(push: bool = True) -> None:
    logger.info("Loading source dataset: %s (rev %s)", SOURCE_DATASET, SOURCE_REVISION)
    ds = load_dataset(
        SOURCE_DATASET,
        data_files={"test": "data/test-*.parquet"},
        split="test",
        verification_mode="no_checks",
        revision=SOURCE_REVISION,
    )
    logger.info("Loaded %d test samples from AVE-Dataset", len(ds))

    query_indices, corpus_indices, query_ids, corpus_ids, qrels_rows = (
        partition_ave_dataset(ds, queries_per_class=4, seed=42)
    )

    logger.info(
        "Partitioned into %d queries and %d corpus documents across %d event classes.",
        len(query_ids),
        len(corpus_ids),
        len(set(ds["label"])),
    )
    logger.info("Total qrels: %d", len(qrels_rows))

    qrels_ds = Dataset.from_list(qrels_rows)

    # For AVE-A2V
    a2v_queries = (
        ds.select(query_indices)
        .add_column("id", query_ids)
        .select_columns(["id", "audio"])
        .cast_column("audio", Audio())
    )
    a2v_corpus = (
        ds.select(corpus_indices)
        .add_column("id", corpus_ids)
        .select_columns(["id", "video"])
    )

    # For AVE-V2A
    v2a_queries = (
        ds.select(query_indices)
        .add_column("id", query_ids)
        .select_columns(["id", "video"])
    )
    v2a_corpus = (
        ds.select(corpus_indices)
        .add_column("id", corpus_ids)
        .select_columns(["id", "audio"])
        .cast_column("audio", Audio())
    )

    logger.info(
        "AVE-A2V datasets prepared: queries=%d, corpus=%d, qrels=%d",
        len(a2v_queries),
        len(a2v_corpus),
        len(qrels_ds),
    )
    logger.info(
        "AVE-V2A datasets prepared: queries=%d, corpus=%d, qrels=%d",
        len(v2a_queries),
        len(v2a_corpus),
        len(qrels_ds),
    )

    if not push:
        logger.info("Skipping HF Hub push as --no-push was specified.")
        return

    try:
        user_info = whoami()
        logger.info("Authenticated as: %s", user_info.get("name"))
        can_write = any(
            "write" in auth
            for auth in user_info.get("auth", {})
            .get("accessToken", {})
            .get("role", "")
            .split(",")
        ) or user_info.get("canPay", False)
    except Exception as e:
        logger.warning("Could not verify HF write credentials: %s", e)
        can_write = False

    try:
        api = HfApi()
        logger.info("Pushing AVE-A2V to %s...", A2V_REPO)
        a2v_queries.push_to_hub(A2V_REPO, config_name="queries", split="test")
        a2v_corpus.push_to_hub(A2V_REPO, config_name="corpus", split="test")
        qrels_ds.push_to_hub(A2V_REPO, config_name="qrels", split="test")
        commit_a2v = api.repo_info(A2V_REPO, repo_type="dataset").sha
        logger.info("Pushed AVE-A2V commit: %s", commit_a2v)

        logger.info("Pushing AVE-V2A to %s...", V2A_REPO)
        v2a_queries.push_to_hub(V2A_REPO, config_name="queries", split="test")
        v2a_corpus.push_to_hub(V2A_REPO, config_name="corpus", split="test")
        qrels_ds.push_to_hub(V2A_REPO, config_name="qrels", split="test")
        commit_v2a = api.repo_info(V2A_REPO, repo_type="dataset").sha
        logger.info("Pushed AVE-V2A commit: %s", commit_v2a)
    except Exception as exc:
        logger.warning(
            "Note: Hugging Face Hub token is read-only or unauthorized for upload: %s\n"
            "Following ColDeRReranking pattern: using mteb/AVE-Dataset with local load_data partition.",
            exc,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create AVE A2V and V2A retrieval datasets"
    )
    parser.add_argument(
        "--push", action="store_true", default=True, help="Push datasets to HF hub"
    )
    parser.add_argument(
        "--no-push",
        dest="push",
        action="store_false",
        help="Do not push datasets to HF hub",
    )
    args = parser.parse_args()

    create_and_push_datasets(push=args.push)
