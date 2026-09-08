from __future__ import annotations

import contextlib
import hashlib
import io
import random
import tempfile
import wave
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
from datasets import Audio, Video, load_dataset

from mteb.abstasks._statistics_calculation import calculate_relevant_docs_statistics
from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.types.statistics import (
    AudioStatistics,
    RetrievalDescriptiveStatistics,
    VideoStatistics,
)

if TYPE_CHECKING:
    from datasets import Dataset

_SOURCE_DATASET = "mteb/AVE-Dataset"
_SOURCE_REVISION = "f6eb93b4e89456277a242583b5565b801bc1981d"

_BIBTEX = r"""
@inproceedings{tian2018audio,
  author = {Tian, Yapeng and Shi, Jing and Li, Bochen and Duan, Zhiyao and Xu, Chenliang},
  booktitle = {Proceedings of the European conference on computer vision (ECCV)},
  pages = {247--263},
  title = {Audio-visual event localization in unconstrained videos},
  year = {2018},
}
"""


def _partition_ave_dataset(
    ds: Dataset, queries_per_class: int = 4, seed: int = 42
) -> tuple[
    list[int],
    list[int],
    list[str],
    list[str],
    dict[str, dict[str, int]],
]:
    """Group AVE-Dataset test clips by label and partition into disjoint queries and corpus."""
    label_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(ds["label"]):
        label_to_indices[label].append(idx)

    rng = random.Random(seed)
    query_indices: list[int] = []
    corpus_indices: list[int] = []
    query_ids: list[str] = []
    corpus_ids: list[str] = []
    query_to_label: dict[str, int] = {}
    corpus_to_label: dict[str, int] = {}

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

    relevant_docs: dict[str, dict[str, int]] = defaultdict(dict)
    for qid, q_label in query_to_label.items():
        for cid, c_label in corpus_to_label.items():
            if q_label == c_label:
                relevant_docs[qid][cid] = 1

    return query_indices, corpus_indices, query_ids, corpus_ids, dict(relevant_docs)


def _compute_audio_stats(raw_ds: Dataset, indices: list[int]) -> AudioStatistics:
    audio_lengths = []
    sampling_rates: dict[int, int] = defaultdict(int)
    hashes = set()
    for idx in indices:
        b = raw_ds[idx]["audio"]["bytes"]
        hashes.add(hashlib.md5(b, usedforsecurity=False).hexdigest())
        with wave.open(io.BytesIO(b)) as w:
            sr = w.getframerate()
            dur = w.getnframes() / sr
            audio_lengths.append(dur)
            sampling_rates[sr] += 1
    return AudioStatistics(
        total_duration_seconds=sum(audio_lengths),
        min_duration_seconds=min(audio_lengths),
        average_duration_seconds=sum(audio_lengths) / len(audio_lengths),
        max_duration_seconds=max(audio_lengths),
        unique_audios=len(hashes),
        average_sampling_rate=sum(k * v for k, v in sampling_rates.items())
        / len(indices),
        sampling_rates=dict(sampling_rates),
    )


def _compute_video_stats(  # noqa: PLR0914
    raw_ds: Dataset, indices: list[int]
) -> VideoStatistics:
    durations, frames_counts, widths, heights = [], [], [], []
    fps_counts: dict[int, int] = defaultdict(int)
    resolution_counts: dict[str, int] = defaultdict(int)
    hashes = set()
    for idx in indices:
        b = raw_ds[idx]["video"]["bytes"]
        hashes.add(hashlib.md5(b, usedforsecurity=False).hexdigest())
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b)
            tmp = f.name
        cap = cv2.VideoCapture(tmp)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dur = frames / fps if fps > 0 else 0.0
        cap.release()
        with contextlib.suppress(OSError):
            Path(tmp).unlink()
        durations.append(dur)
        frames_counts.append(frames)
        widths.append(w)
        heights.append(h)
        fps_counts[round(fps)] += 1
        resolution_counts[f"{w}x{h}"] += 1

    min_w, max_w = min(widths), max(widths)
    min_h, max_h = min(heights), max(heights)
    avg_w = sum(widths) / len(widths)
    avg_h = sum(heights) / len(heights)

    return VideoStatistics(
        total_duration_seconds=sum(durations),
        total_frames=sum(frames_counts),
        min_width=min_w,
        average_width=avg_w,
        max_width=max_w,
        min_height=min_h,
        average_height=avg_h,
        max_height=max_h,
        min_duration_seconds=min(durations),
        average_duration_seconds=sum(durations) / len(durations),
        max_duration_seconds=max(durations),
        unique_videos=len(hashes),
        average_fps=sum(fps * count for fps, count in fps_counts.items())
        / sum(fps_counts.values()),
        fps=dict(fps_counts),
        min_resolution=(min_w, min_h),
        average_resolution=(avg_w, avg_h),
        max_resolution=(max_w, max_h),
        resolutions=dict(resolution_counts),
    )


class AVEA2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AVEA2VRetrieval",
        description=(
            "Audio-to-video event retrieval on the AVE dataset (Closes #5413). "
            "Given an audio recording, retrieve corresponding video clips depicting "
            "the same event category across 28 event classes."
        ),
        reference="https://arxiv.org/abs/1807.03960",
        dataset={
            "path": _SOURCE_DATASET,
            "revision": _SOURCE_REVISION,
        },
        type="Any2AnyRetrieval",
        category="a2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2018-01-01", "2018-09-01"),
        domains=["AudioScene", "Web"],
        task_subtypes=["Cross-Modal Retrieval", "Environment Sound Classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio", "video"],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Retrieve video clips of the event heard in this audio recording."
        },
        is_beta=False,
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        ds = load_dataset(
            self.metadata.dataset["path"],
            data_files={"test": "data/test-*.parquet"},
            split="test",
            verification_mode="no_checks",
            revision=self.metadata.dataset.get("revision"),
        )
        query_indices, corpus_indices, query_ids, corpus_ids, relevant_docs = (
            _partition_ave_dataset(ds)
        )

        queries_ds = (
            ds.select(query_indices)
            .add_column("id", query_ids)
            .select_columns(["id", "audio"])
            .cast_column("audio", Audio())
        )
        corpus_ds = (
            ds.select(corpus_indices)
            .add_column("id", corpus_ids)
            .select_columns(["id", "video"])
            .cast_column("video", Video())
        )

        self.dataset = {
            "default": {
                "test": RetrievalSplitData(
                    queries=queries_ds,
                    corpus=corpus_ds,
                    relevant_docs=relevant_docs,
                    top_ranked=None,
                )
            }
        }
        self.data_loaded = True

    def _calculate_descriptive_statistics_from_split(
        self,
        split: str,
        *,
        hf_subset: str | None = None,
        compute_overall: bool = False,
        num_proc: int | None = None,
    ) -> RetrievalDescriptiveStatistics:
        if not self.data_loaded:
            self.load_data(num_proc=num_proc)

        split_data = self.dataset["default"][split]
        relevant_docs = split_data["relevant_docs"]

        ds = load_dataset(
            self.metadata.dataset["path"],
            data_files={"test": "data/test-*.parquet"},
            split="test",
            verification_mode="no_checks",
            revision=self.metadata.dataset.get("revision"),
        )
        raw_ds = ds.cast_column("audio", Audio(decode=False)).cast_column(
            "video", Video(decode=False)
        )
        query_indices, corpus_indices, query_ids, corpus_ids, _ = (
            _partition_ave_dataset(raw_ds)
        )

        rel_stats = calculate_relevant_docs_statistics(
            relevant_docs, set(query_ids), set(corpus_ids)
        )
        q_audio = _compute_audio_stats(raw_ds, query_indices)
        c_video = _compute_video_stats(raw_ds, corpus_indices)

        return RetrievalDescriptiveStatistics(
            num_samples=len(query_ids) + len(corpus_ids),
            num_queries=len(query_ids),
            num_documents=len(corpus_ids),
            number_of_characters=0,
            documents_text_statistics=None,
            documents_image_statistics=None,
            documents_audio_statistics=None,
            documents_video_statistics=c_video,
            queries_text_statistics=None,
            queries_image_statistics=None,
            queries_audio_statistics=q_audio,
            queries_video_statistics=None,
            relevant_docs_statistics=rel_stats,
            top_ranked_statistics=None,
        )


class AVEV2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AVEV2ARetrieval",
        description=(
            "Video-to-audio event retrieval on the AVE dataset (Closes #5413). "
            "Given a video clip, retrieve corresponding audio recordings depicting "
            "the same event category across 28 event classes."
        ),
        reference="https://arxiv.org/abs/1807.03960",
        dataset={
            "path": _SOURCE_DATASET,
            "revision": _SOURCE_REVISION,
        },
        type="Any2AnyRetrieval",
        category="v2a",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2018-01-01", "2018-09-01"),
        domains=["AudioScene", "Web"],
        task_subtypes=["Cross-Modal Retrieval", "Environment Sound Classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio"],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Retrieve audio recordings of the event seen in this video."},
        is_beta=False,
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        ds = load_dataset(
            self.metadata.dataset["path"],
            data_files={"test": "data/test-*.parquet"},
            split="test",
            verification_mode="no_checks",
            revision=self.metadata.dataset.get("revision"),
        )
        query_indices, corpus_indices, query_ids, corpus_ids, relevant_docs = (
            _partition_ave_dataset(ds)
        )

        queries_ds = (
            ds.select(query_indices)
            .add_column("id", query_ids)
            .select_columns(["id", "video"])
            .cast_column("video", Video())
        )
        corpus_ds = (
            ds.select(corpus_indices)
            .add_column("id", corpus_ids)
            .select_columns(["id", "audio"])
            .cast_column("audio", Audio())
        )

        self.dataset = {
            "default": {
                "test": RetrievalSplitData(
                    queries=queries_ds,
                    corpus=corpus_ds,
                    relevant_docs=relevant_docs,
                    top_ranked=None,
                )
            }
        }
        self.data_loaded = True

    def _calculate_descriptive_statistics_from_split(
        self,
        split: str,
        *,
        hf_subset: str | None = None,
        compute_overall: bool = False,
        num_proc: int | None = None,
    ) -> RetrievalDescriptiveStatistics:
        if not self.data_loaded:
            self.load_data(num_proc=num_proc)

        split_data = self.dataset["default"][split]
        relevant_docs = split_data["relevant_docs"]

        ds = load_dataset(
            self.metadata.dataset["path"],
            data_files={"test": "data/test-*.parquet"},
            split="test",
            verification_mode="no_checks",
            revision=self.metadata.dataset.get("revision"),
        )
        raw_ds = ds.cast_column("audio", Audio(decode=False)).cast_column(
            "video", Video(decode=False)
        )
        query_indices, corpus_indices, query_ids, corpus_ids, _ = (
            _partition_ave_dataset(raw_ds)
        )

        rel_stats = calculate_relevant_docs_statistics(
            relevant_docs, set(query_ids), set(corpus_ids)
        )
        q_video = _compute_video_stats(raw_ds, query_indices)
        c_audio = _compute_audio_stats(raw_ds, corpus_indices)

        return RetrievalDescriptiveStatistics(
            num_samples=len(query_ids) + len(corpus_ids),
            num_queries=len(query_ids),
            num_documents=len(corpus_ids),
            number_of_characters=0,
            documents_text_statistics=None,
            documents_image_statistics=None,
            documents_audio_statistics=c_audio,
            documents_video_statistics=None,
            queries_text_statistics=None,
            queries_image_statistics=None,
            queries_audio_statistics=None,
            queries_video_statistics=q_video,
            relevant_docs_statistics=rel_stats,
            top_ranked_statistics=None,
        )
