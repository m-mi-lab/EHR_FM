import abc
import pickle
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path

import polars as pl
import torch

from safetensors.torch import save_file
import sys
sys.path.append("..")
from ..patterns import MatchAndRevise
from ..constants import SpecialToken as ST
from ..constants import STATIC_DATA_FN
from ..vocabulary import Vocabulary
from ._sharded_data import ShardedData



class TimelineDataset(torch.utils.data.Dataset):
    def __init__(
        self, input_dir: str | Path, n_positions: int = 2048, is_encoder_decoder: bool = False
    ):
        input_dir = Path(input_dir)
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {input_dir}")

        self._data = ShardedData(input_dir)

        self.vocab = Vocabulary.from_path(input_dir)
        self._num_quantiles = len(self.vocab.quantile_stokens)
        self.static_data = pickle.load((input_dir / STATIC_DATA_FN).open("rb"))

        # plus one, because DOB takes 2 spots
        self.context_size = len(next(iter(self.static_data.values()))) + 1
        self.timeline_size = n_positions - self.context_size

        self.is_encoder_decoder = is_encoder_decoder
        if is_encoder_decoder:
            self.timeline_size = n_positions

        self.cxr_tokens = torch.tensor(
            self.vocab.encode([stoken for stoken in self.vocab if stoken.startswith("CXR//")])
        )

    @property
    def tokens(self):
        return self._data.tokens

    @property
    def times(self):
        return self._data.times

    @property
    def patient_ids(self) -> torch.Tensor:
        return torch.cat([shard["patient_ids"] for shard in self._data.shards])

    @property
    def patient_id_at_idx(self):
        return self._data.patient_id_at_idx

    @property
    def patient_offsets(self) -> list[torch.Tensor]:
        return torch.cat([shard["patient_offsets"] + shard["offset"] for shard in self._data.shards])

    @property
    def patient_offset_at_idx(self):
        """Aka patient data start at idx."""
        return self._data.patient_offset_at_idx

    @property
    def patient_data_end_at_idx(self):
        return self._data.patient_data_end_at_idx

    @property
    def is_mimic(self):
        return "hadm_id" in self._data.shards[0]

    @property
    def hadm_id(self):
        if not self.is_mimic:
            raise AttributeError("It's not MIMIC, no 'hadm_id' available.")
        return self._data.hadm_id

    @property
    def icu_stay_id(self):
        if not self.is_mimic:
            raise AttributeError("It's not MIMIC, no 'icustay_id' available.")
        return self._data.icu_stay_id

    @property
    def dicom_id(self):
        if not self.is_mimic:
            raise AttributeError("It's not MIMIC with CXR extension, no 'dicom_id' available.")
        return self._data.dicom_id

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(len={len(self):,}, "
            f"patient_num={len(self.patient_ids):,}, "
            f"vocab_size={len(self.vocab):,})"
        )

    def __len__(self) -> int:
        return len(self.tokens) - self.timeline_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor | tuple, torch.Tensor]:
        pt_ctx = self._get_patient_context(idx)
        timeline = self.tokens[idx : idx + self.timeline_size + 1]

        if self.is_encoder_decoder:
            return (pt_ctx, timeline[:-1]), timeline[1:]

        cxr_ids = None
        if self.cxr_tokens.numel() and torch.any(cxr_token_mask := torch.isin(timeline, self.cxr_tokens)):
            cxr_token_indices = torch.nonzero(cxr_token_mask).view(-1)
            cxr_ids = torch.tensor(
                [
                    self.dicom_id[idx + cxr_idx]
                    for i, cxr_idx in enumerate(cxr_token_indices, 1)
                    if cxr_idx < len(timeline) - i
                ]
            ).to(int)

            new_timeline, ptr, inserted_cxr_indices = [], 0, []
            for i, (cxr_idx, cxr_id) in enumerate(zip(cxr_token_indices, cxr_ids)):
                new_timeline.append(timeline[ptr : cxr_idx + 2].clone())
                new_timeline[-1][-1] = cxr_id
                ptr = cxr_idx + 1
                inserted_cxr_indices.append(ptr.item() + i)
            new_timeline.append(timeline[ptr : len(timeline) - len(cxr_ids)])
            timeline = torch.cat(new_timeline)

        x = torch.cat((pt_ctx, timeline[:-1]))

        if cxr_ids is not None and cxr_ids.numel():
            timeline[inserted_cxr_indices] = -100

        y = torch.cat((pt_ctx, timeline[1:]))
        y[: self.context_size] = -100
        return x, y

    def _get_patient_context(self, idx: int) -> torch.Tensor:
        patient_id = self.patient_id_at_idx[idx].item()
        time_at_start = self.times[idx].item()

        static_tokens = []
        if patient_id not in self.static_data:
            # TODO: Don't hardcode this.
            static_tokens.extend(
                [
                    "BMI//UNKNOWN",
                    "GENDER//M",
                    "MARITAL//UNKNOWN",
                    f"Q{int(self._num_quantiles / 2)}",
                    "Q1",
                    "RACE//UNKNOWN",
                ]
            )
        else:
            for token in self.static_data[patient_id].values():
                if token["code"][0] == ST.DOB:
                    age = timedelta(microseconds=time_at_start - token["time"][0])
                    static_tokens.extend(self._age_to_tokens(age.days / 365.25))
                elif len(token["code"]) == 1:
                    static_tokens.append(token["code"][0])
                else:
                    idx = self._find_closest_index(token["time"], time_at_start)
                    static_tokens.append(token["code"][idx])
        return torch.tensor(self.vocab.encode(static_tokens))

    def _age_to_tokens(self, age_years: float) -> tuple[str]:
        age_scaled = age_years * self._num_quantiles**2 / 100
        age_scaled = min(age_scaled, self._num_quantiles**2 - 1)

        age_t1 = int(age_scaled // self._num_quantiles)
        age_t2 = round(age_scaled % self._num_quantiles)
        if age_t2 == self._num_quantiles:
            age_t1 += 1
            age_t2 = 0

        return f"Q{age_t1 + 1}", f"Q{age_t2 + 1}"

    @staticmethod
    def _find_closest_index(ll: list, target_value: float) -> int:
        return min(range(len(ll)), key=lambda i: abs(ll[i] - target_value))

    @staticmethod
    def tensorize(in_fp: str | Path | list, out_fp: str | Path, vocab: Vocabulary):
        df = (
            pl.scan_parquet(in_fp)
            .with_columns(
                tokens=pl.col("code").replace_strict(vocab.stoi, default=None, return_dtype=pl.Int64),
                times=pl.col("time").cast(pl.Int64),
            )
            # Filter out negative times, dates before 1970 are most likely errors
            .filter(pl.col("time") >= 0)
            .collect()
        )

        patient_id_col = MatchAndRevise.sort_cols[0]
        patient_df = (
            df.set_sorted(patient_id_col)
            .group_by(patient_id_col, maintain_order=True)
            .agg(len=pl.len())
            .select(patient_id_col, offsets=pl.col("len").cum_sum().shift(1, fill_value=0))
        )
        tensors = {
            "tokens": df["tokens"].to_torch(),
            "times": df["times"].to_torch(),
            "patient_ids": patient_df[patient_id_col].to_torch(),
            "patient_offsets": patient_df["offsets"].to_torch(),
        }

        # TODO: This is extremely memory-inefficient.
        for mimic_col in ["hadm_id", "icustay_id", "dicom_id"]:
            if mimic_col in df.columns:
                tensors[mimic_col] = df[mimic_col].to_torch()

        save_file(tensors, Path(out_fp).with_suffix(".safetensors"))


class InferenceDataset(TimelineDataset, abc.ABC):
    # INFERENCE DEFAULT CONSTRAINTS
    stop_stokens: list[ST] = [ST.DEATH, ST.TIMELINE_END]  # Default inference stop tokens
    time_limit: timedelta = timedelta(days=365.25 * 2)  # Inference time constraint

    def _get_hadm_id(self, idx: int) -> int | None:
        return None if torch.isnan(hadm_id := self.hadm_id[idx]) else int(hadm_id)

    def _get_icu_stay_id(self, idx: int) -> int | None:
        return None if torch.isnan(icu_stay_id := self.icu_stay_id[idx]) else int(icu_stay_id)

    def _get_dicom_id(self, idx: int) -> str | None:
        return None if torch.isnan(dicom_id := self.dicom_id[idx]) else dicom_id

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    def __getitem__(self, idx: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        data_start_idx = self.patient_offset_at_idx[idx]
        if idx - data_start_idx + 1 > self.timeline_size:
            data_start_idx = idx + 1 - self.timeline_size

        pt_ctx = self._get_patient_context(data_start_idx)
        timeline = self.tokens[data_start_idx : idx + 1]
        if self.is_encoder_decoder:
            return (pt_ctx, timeline)
        return torch.cat((pt_ctx, timeline))

    def _get_indices_of_stokens(self, stokens: str | Sequence[str]) -> torch.Tensor:
        if isinstance(stokens, str):
            stokens = [stokens]
        tokens_of_interest = torch.tensor(self.vocab.encode(stokens))
        shard_indices = []
        token_offset = 0
        for token_chunk in self.tokens:
            new_indices = torch.nonzero(torch.isin(token_chunk, tokens_of_interest)).view(-1)
            new_indices += token_offset
            shard_indices.append(new_indices)
            token_offset += len(token_chunk)

        return torch.cat(shard_indices)

    def _match(
        self,
        ordered_sequence: torch.Tensor,
        input: torch.Tensor,
        *,
        fill_unmatched: int | None = None,
        shift: int = 0,
    ) -> torch.Tensor:
        """TODO: Write a docstring, because this function is hell."""
        ordered_sequence_indices = torch.searchsorted(ordered_sequence, input, right=True)
        if shift:
            ordered_sequence_indices += shift
        if fill_unmatched is None:
            return ordered_sequence[ordered_sequence_indices]
        else:
            out = torch.full_like(input, fill_value=fill_unmatched)
            mask = ordered_sequence_indices < len(ordered_sequence)
            out[mask] = ordered_sequence[ordered_sequence_indices[mask]]
            return out
