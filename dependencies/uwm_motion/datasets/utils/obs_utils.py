from typing import Union

import torch


def unflatten_obs(x: dict[str, torch.Tensor]) -> dict[str, Union[dict, torch.Tensor]]:
    """
    Unflattens entries with keys starting with "obs" as follows:

        dict_data["obs.some_name"] -> dict_data["obs"]["some_name"]

    Args:
        x: A dictionary of tensors.

    Returns:
        The same dictionary, but keys starting with "obs" are unflattened.
    """

    obs = {}
    to_delete = []
    for key, value in x.items():
        if key.startswith("obs."):
            new_key = key[4:]
            assert new_key not in obs, f"Duplicate key {new_key}"
            obs[new_key] = value
            to_delete.append(key)
    for key in to_delete:
        del x[key]
    x["obs"] = obs
    return x
