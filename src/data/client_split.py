from __future__ import annotations

import random
from dataclasses import dataclass

from .loaders import Example, load_task_examples


@dataclass
class ClientData:
    client_id: str
    task: str
    group: str
    train: list[Example]
    val: list[Example]
    test: list[Example]


def _slice(values: list[Example], start: int, n_train: int, n_val: int, n_test: int):
    needed = n_train + n_val + n_test
    if start + needed > len(values):
        raise ValueError(f"Not enough examples: need index {start + needed}, have {len(values)}")
    part = values[start:start + needed]
    return part[:n_train], part[n_train:n_train + n_val], part[n_train + n_val:]


def _apply_sample_cap(clients: list[ClientData], cap) -> list[ClientData]:
    if cap is None:
        return clients
    if isinstance(cap, dict):
        limits = {split: int(cap.get(split, 10**9)) for split in ["train", "val", "test"]}
    else:
        limits = {split: int(cap) for split in ["train", "val", "test"]}
    if any(value <= 0 for value in limits.values()):
        raise ValueError("max_samples_per_client limits must be positive")
    for client in clients:
        client.train = client.train[:limits["train"]]
        client.val = client.val[:limits["val"]]
        client.test = client.test[:limits["test"]]
    return clients


def build_clients(cfg: dict, seed: int) -> list[ClientData]:
    tasks = list(cfg.get("tasks", ["object_counting"]))
    k = int(cfg.get("num_clients", 3))
    n_train = int(cfg.get("train_per_client", 3))
    n_val = int(cfg.get("val_per_client", 2))
    n_test = int(cfg.get("test_per_client", 3))
    per_client = n_train + n_val + n_test
    setting = cfg.get("setting", "task_skew")
    loaded = {task: load_task_examples(task, cfg, seed + idx * 997) for idx, task in enumerate(tasks)}
    official = cfg.get("source") == "official_fedtextgrad"
    if official:
        partitions = {
            task: {
                split: [example for example in values if example.split == split]
                for split in ["train", "val", "test"]
            }
            for task, values in loaded.items()
        }
        for task_idx, task in enumerate(tasks):
            for split_idx, split in enumerate(["train", "val", "test"]):
                random.Random(seed + task_idx * 997 + split_idx * 101).shuffle(partitions[task][split])
    clients = []

    if setting == "iid":
        task = tasks[0]
        for i in range(k):
            if official:
                train = partitions[task]["train"][i * n_train:(i + 1) * n_train]
                val = partitions[task]["val"][i * n_val:(i + 1) * n_val]
                test = partitions[task]["test"][i * n_test:(i + 1) * n_test]
                if min(len(train), len(val), len(test)) == 0:
                    raise ValueError("Official split is too small for the requested IID client construction")
            else:
                pool = loaded[task]
                train, val, test = _slice(pool, i * per_client, n_train, n_val, n_test)
            clients.append(ClientData(f"client_{i}", task, task, train, val, test))
        return _apply_sample_cap(clients, cfg.get("max_samples_per_client"))

    offsets = {task: 0 for task in tasks}
    official_offsets = {task: {"train": 0, "val": 0, "test": 0} for task in tasks}
    official_used = {task: {"train": set(), "val": set(), "test": set()} for task in tasks}
    for i in range(k):
        task = tasks[i % len(tasks)]
        pool = loaded[task]
        if setting == "mixed":
            # Task skew plus client-specific difficulty strata.
            parity = (i // len(tasks)) % 2
            if not official:
                pool = sorted(pool, key=lambda x: x.difficulty, reverse=bool(parity))
            group = f"{task}:{'hard' if parity else 'easy'}"
        elif setting == "task_skew":
            group = task
        else:
            raise ValueError(f"Unknown distribution setting: {setting}")
        if official:
            selected = {}
            for split, count in [("train", n_train), ("val", n_val), ("test", n_test)]:
                split_pool = partitions[task][split]
                if setting == "mixed":
                    split_pool = sorted(split_pool, key=lambda x: x.difficulty, reverse=bool(parity))
                    # Difficulty order changes between easy/hard clients.  A
                    # numeric offset into the reversed list can therefore reuse
                    # examples selected from the forward list.  Select the
                    # first unused examples in the requested difficulty order.
                    selected[split] = [
                        example for example in split_pool
                        if example.id not in official_used[task][split]
                    ][:count]
                    official_used[task][split].update(example.id for example in selected[split])
                else:
                    start = official_offsets[task][split]
                    selected[split] = split_pool[start:start + count]
                    official_offsets[task][split] += count
                if len(selected[split]) != count:
                    raise ValueError(f"Official {task}/{split} split is too small for requested clients")
            train, val, test = selected["train"], selected["val"], selected["test"]
        else:
            start = offsets[task]
            offsets[task] += per_client
            train, val, test = _slice(pool, start, n_train, n_val, n_test)
        clients.append(ClientData(f"client_{i}", task, group, train, val, test))

    random.Random(seed).shuffle(clients)
    clients.sort(key=lambda c: int(c.client_id.split("_")[-1]))
    return _apply_sample_cap(clients, cfg.get("max_samples_per_client"))
