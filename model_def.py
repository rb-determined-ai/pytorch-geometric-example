from typing import Any, Dict, Sequence, Tuple, Union, cast

import torch
from torch import nn

import determined as det
from determined import pytorch

class OnesDataset(torch.utils.data.Dataset):
    # TODO: override this with the some dummy pytorch geometric data
    def __len__(self) -> int:
        return 64

    def __getitem__(self, index: int) -> Tuple:
        return torch.Tensor([1.0])


class ExamplePytorchTrial(pytorch.PyTorchTrial):
    # TODO: override this with the pytorch geometric model
    def __init__(self, context):
        self.context = context

        self.model = context.wrap_model(nn.Linear(1, 1, False))
        # initialize weights to 0
        self.model.weight.data.fill_(0)
        self.opt = context.wrap_optimizer(
            torch.optim.SGD(self.model.parameters(), lr=0.001), backward_passes_per_step=2
        )

    def train_batch(
        self, batch: pytorch.TorchData, epoch_idx: int, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        loss = torch.nn.MSELoss()(self.model(batch), batch)
        self.context.backward(loss, retain_graph=True)
        self.context.backward(loss)
        print(f"loss: {loss} at {batch_idx} with weight {self.model.weight.data[0]}")
        self.context.step_optimizer(self.opt)
        return {"loss": loss, "d": {"metric": "asdf"}}

    def evaluate_batch(self, batch: pytorch.TorchData) -> Dict[str, Any]:
        data = labels = batch
        loss = torch.nn.MSELoss()(self.model(data), labels)
        return {"loss": loss}

    def build_training_data_loader(self):
        # TODO: use Determined's Sampler API + pytorch_geomtric's DataLoader
        return pytorch.DataLoader(
            OnesDataset(), batch_size=self.context.get_per_slot_batch_size()
        )

    def build_validation_data_loader(self):
        # TODO: use Determined's Sampler API + pytorch_geomtric's DataLoader
        return pytorch.DataLoader(
            OnesDataset(), batch_size=self.context.get_per_slot_batch_size()
        )


    def _records_in_batch(self, batch):
        """Count the number of records batch.  Only needs overriding for unusal datasets."""
        # TODO: override this for pytorch_geometric batch objects
        return det.pytorch.data_length(batch)

    def _batch_to_device(self, batch, context):
        """Move a batch to the model.  Only needs overriding for unusal datasets."""
        # TODO: override this for pytorch_geometric batch objects
        return context.to_device(batch)
