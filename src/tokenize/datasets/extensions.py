import torch as th

from ethos.datasets.base import InferenceDataset


def create_single_trajectory_dataset(dataset_cls: type[InferenceDataset]) -> type[InferenceDataset]:
    class SingleTrajectoryDataset(InferenceDataset):
        def __init__(self, sample_idx: int, *args, **kwargs):
            super().__init__(*args, **kwargs)

            dataset = dataset_cls(*args, **kwargs)

            self.stop_stokens = dataset.stop_stokens
            # could be adapted to the time offset it starts the prediction from
            self.time_limit = dataset.time_limit

            _, y = dataset[sample_idx]
            self.start_idx = y["data_idx"]
            self.outcome_idx = self.start_idx + y["true_token_dist"]
            self.y = {
                "expected": y["expected"],
                "patient_id": y["patient_id"],
            }

        def __len__(self) -> int:
            return self.outcome_idx - self.start_idx

        def __getitem__(self, idx) -> tuple[th.Tensor, dict]:
            start_idx = self.start_idx + int(idx)
            return super().__getitem__(start_idx), {
                **self.y,
                "start_token": self.vocab.decode(self.tokens[start_idx]),
                "start_time": self.times[start_idx].item(),
                "true_token_dist": self.outcome_idx - start_idx,
                "true_token_time": (self.times[self.outcome_idx] - self.times[start_idx]).item(),
                "data_idx": start_idx,
            }

    return SingleTrajectoryDataset
