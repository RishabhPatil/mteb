from __future__ import annotations

import random
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from datasets import concatenate_datasets, load_dataset

if TYPE_CHECKING:
    from datasets import Dataset

from mteb.abstasks import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


def _generate_pairs(
    class_labels: list,
    rng: random.Random,
    max_per_side: int = 1024,
) -> list[tuple[int, int, int]]:
    label_groups: dict[object, list[int]] = defaultdict(list)
    for i, label in enumerate(class_labels):
        label_groups[label].append(i)

    all_labels = list(label_groups.keys())
    pos_pairs: list[tuple[int, int]] = []
    neg_pairs: list[tuple[int, int]] = []
    indices = list(range(len(class_labels)))
    rng.shuffle(indices)

    for i in indices:
        cls = class_labels[i]
        same = [j for j in label_groups[cls] if j != i]
        if same and len(pos_pairs) < max_per_side:
            pos_pairs.append((i, rng.choice(same)))
        others = [label for label in all_labels if label != cls]
        if others and len(neg_pairs) < max_per_side:
            neg_cls = rng.choice(others)
            neg_pairs.append((i, rng.choice(label_groups[neg_cls])))
        if len(pos_pairs) >= max_per_side and len(neg_pairs) >= max_per_side:
            break

    n = min(len(pos_pairs), len(neg_pairs))
    pairs: list[tuple[int, int, int]] = [(a, b, 1) for a, b in pos_pairs[:n]]
    pairs += [(a, b, 0) for a, b in neg_pairs[:n]]
    rng.shuffle(pairs)
    return pairs


def _build_pair_dataset(
    ds: Dataset,
    pairs: list[tuple[int, int, int]],
) -> Dataset:
    idx1 = [p[0] for p in pairs]
    idx2 = [p[1] for p in pairs]
    labels = [p[2] for p in pairs]

    chunk_size = 64
    chunks: list[Dataset] = []
    for start in range(0, len(pairs), chunk_size):
        end = min(start + chunk_size, len(pairs))
        d1 = (
            ds.select(idx1[start:end])
            .select_columns(["video"])
            .rename_columns({"video": "video1"})
        )
        d2 = (
            ds.select(idx2[start:end])
            .select_columns(["video"])
            .rename_columns({"video": "video2"})
        )
        chunk = concatenate_datasets([d1, d2], axis=1)
        chunk = chunk.add_column("label", labels[start:end])
        chunks.append(chunk)

    return concatenate_datasets(chunks) if len(chunks) > 1 else chunks[0]


class HMDB51VideoPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="HMDB51VideoPairClassification",
        description=(
            "Pair classification on the HMDB51 dataset: determining whether two video "
            "clips depict the same human action category across 51 classes. Balanced "
            "same-class and different-class pairs are sampled with fixed seed 42 from "
            "official split 1 (1,530 test clips). Closes #5412."
        ),
        reference="https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/",
        dataset={
            "path": "mteb/HMDB51",
            "revision": "7f9af5438a855e9348fb23ecb5ec740a9c21daf3",
        },
        type="VideoPairClassification",
        category="v2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2011-01-01", "2011-12-31"),
        domains=["Activity", "Web"],
        task_subtypes=["Activity recognition"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{6126543,
  author = {Kuehne, H. and Jhuang, H. and Garrote, E. and Poggio, T. and Serre, T.},
  booktitle = {2011 International Conference on Computer Vision},
  doi = {10.1109/ICCV.2011.6126543},
  keywords = {Cameras;YouTube;Databases;Training;Visualization;Humans;Motion pictures},
  number = {},
  pages = {2556-2563},
  title = {HMDB: A large video database for human motion recognition},
  volume = {},
  year = {2011},
}
""",
        is_beta=False,
    )

    input1_column_name: str = "video1"
    input2_column_name: str = "video2"
    label_column_name: str = "label"

    def load_data(self, **kwargs: Any) -> None:
        if self.data_loaded:
            return
        path = self.metadata.dataset["path"]
        revision = self.metadata.dataset["revision"]
        ds = load_dataset(
            path,
            revision=revision,
            data_files={"test": "data/test-*.parquet"},
            split="test",
            verification_mode="no_checks",
        )
        pairs = _generate_pairs(ds["label"], random.Random(42), max_per_side=1024)
        paired_dataset = _build_pair_dataset(ds, pairs)
        self.dataset = {"test": paired_dataset}
        self.data_loaded = True
