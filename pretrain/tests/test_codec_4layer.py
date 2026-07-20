import torch

from openonerec_vllm085_residual_sid.codec import (
    pack_candidates,
    unpack_data,
)


def test_four_layer_round_trip():
    ids = torch.tensor(
        [[[10, 20, 30, 40], [11, 21, 31, 41]]],
        dtype=torch.long,
    )
    scores = torch.tensor([[-1.25, -2.50]])
    packed = pack_candidates(ids, scores)
    assert tuple(packed.shape) == (1, 2, 5)

    candidates = unpack_data(
        packed[0],
        beam_size=2,
        num_layers=4,
    )
    assert candidates[0].global_ids == (10, 20, 30, 40)
    assert candidates[1].global_ids == (11, 21, 31, 41)
    assert abs(candidates[0].score + 1.25) < 1e-6
