import random
from collections import defaultdict

from torch.utils.data import Sampler


class RandomIdentitySampler(Sampler):
    """P x K sampling for triplet loss: each batch holds batch_size // num_instances identities
    with num_instances images each, so every anchor has at least one positive in its batch.
    Identities with fewer than num_instances images are padded by repeating their images.
    """

    def __init__(self, labels, batch_size, num_instances=2):
        assert batch_size % num_instances == 0, "batch_size must be a multiple of num_instances"
        self.num_instances = num_instances
        self.ids_per_batch = batch_size // num_instances
        self.index_by_id = defaultdict(list)
        for idx, label in enumerate(labels):
            self.index_by_id[label].append(idx)
        self.ids = list(self.index_by_id)
        self.length = (len(self.ids) // self.ids_per_batch) * batch_size

    def __iter__(self):
        chunks = {}
        for pid, idxs in self.index_by_id.items():
            idxs = list(idxs)   
            if len(idxs) < self.num_instances:
                idxs += random.choices(idxs, k=self.num_instances - len(idxs))
            random.shuffle(idxs)
            usable = len(idxs) - len(idxs) % self.num_instances
            chunks[pid] = [idxs[i:i + self.num_instances] for i in range(0, usable, self.num_instances)]

        order = []
        available = [pid for pid in self.ids if chunks[pid]]
        while len(available) >= self.ids_per_batch:
            for pid in random.sample(available, self.ids_per_batch):
                order.extend(chunks[pid].pop())
            available = [pid for pid in available if chunks[pid]]
        return iter(order)

    def __len__(self):
        return self.length
