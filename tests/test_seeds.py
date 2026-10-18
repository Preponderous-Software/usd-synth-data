import numpy as np
import pytest

from usd_synth.seeds import STAGES, render_seed, scene_seed, stage_key, stage_rng

# Golden values: changing any of these silently changes every existing dataset. A change here
# must be deliberate and called out (ARCHITECTURE.md, "Determinism and seeds").
GOLDEN = {
    (42, 0, "objects"): [396612754, 3106600049, 805520540],
    (42, 137, "poses"): [464974636, 3535488266, 1839892590],
    (0, 0, "camera"): [1148376675, 1249199844, 2559232306],
    (7, 3, "render"): [3142031167, 125310125, 266018602],
}


@pytest.mark.parametrize("key", sorted(GOLDEN))
def test_golden_first_draws(key):
    assert stage_rng(*key).integers(0, 2**32, 3).tolist() == GOLDEN[key]


def test_golden_scene_and_render_seed():
    assert scene_seed(42, 137) == 7519799727731610501
    assert render_seed(42, 137) == 1402544911


def test_stage_keys_are_distinct_and_fixed():
    keys = [stage_key(s) for s in STAGES]
    assert len(set(keys)) == len(keys)
    assert stage_key("objects") == 2988100851


def test_unknown_stage_rejected():
    with pytest.raises(ValueError, match="unknown stage"):
        stage_rng(1, 0, "physics")


def test_negative_inputs_rejected():
    with pytest.raises(ValueError):
        stage_rng(-1, 0, "objects")
    with pytest.raises(ValueError):
        stage_rng(1, -1, "objects")


def test_extra_draws_in_one_stage_do_not_shift_another():
    baseline = stage_rng(42, 5, "camera").random(4)
    greedy = stage_rng(42, 5, "objects")
    greedy.random(10_000)  # a stage that suddenly consumes far more numbers
    after = stage_rng(42, 5, "camera").random(4)
    np.testing.assert_array_equal(baseline, after)


def test_streams_independent_of_generation_order():
    forward = [stage_rng(9, i, "poses").random() for i in range(5)]
    backward = [stage_rng(9, i, "poses").random() for i in reversed(range(5))][::-1]
    assert forward == backward


def test_streams_differ_by_seed_index_and_stage():
    a = stage_rng(1, 0, "objects").random()
    assert a != stage_rng(2, 0, "objects").random()
    assert a != stage_rng(1, 1, "objects").random()
    assert a != stage_rng(1, 0, "poses").random()
