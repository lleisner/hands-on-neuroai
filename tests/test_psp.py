from __future__ import annotations

import torch

from hands_on_neuroai.models.psp import PSPLinear
from hands_on_neuroai.models.context import (
    generate_binary_context,
    generate_complex_context,
    generate_rotation_context,
    get_context_generator,
)


def test_psp_linear_matches_linear_with_ones_context():
    """If PSPLinear is given a context_fn that always returns ones,
    it should behave exactly like a plain nn.Linear.
    """

    def ones_context(size: int, seed=None):
        return torch.ones(size)

    in_features = 8
    out_features = 4
    num_tasks = 1
    seed = 42

    base_linear = torch.nn.Linear(in_features, out_features)
    psp = PSPLinear(
        in_features,
        out_features,
        num_tasks=num_tasks,
        context_fn=ones_context,
        seed=seed,
    )

    # copy weights so they match exactly
    psp.linear.weight.data.copy_(base_linear.weight.data)
    psp.linear.bias.data.copy_(base_linear.bias.data)

    # set task 0 (the only one)
    psp.set_task(0)

    x = torch.randn(5, in_features)
    y_base = base_linear(x)
    y_psp = psp(x)

    assert torch.allclose(y_base, y_psp)


def test_generate_binary_context_values():
    ctx = generate_binary_context(16, seed=123)
    unique_vals = torch.unique(ctx)
    assert set(unique_vals.tolist()) == {-1.0, 1.0}


def test_generate_complex_context_shape_and_unit_norm():
    ctx = generate_complex_context(10, seed=0)
    # shape should be (size, 2)
    assert ctx.shape == (10, 2)

    # check unit norm of each vector: cos^2 + sin^2 = 1
    norms = torch.sqrt(ctx[:, 0] ** 2 + ctx[:, 1] ** 2)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_generate_rotation_context_shape_and_unit_norm():
    ctx = generate_rotation_context(32, seed=5)
    assert ctx.shape == (32,)
    assert torch.allclose(ctx.norm(p=2), torch.tensor(1.0), atol=1e-6)


def test_psp_linear_has_different_contexts_per_task():
    """Ensure PSPLinear stores distinct contexts per task and switches them properly."""
    in_features = 6
    out_features = 3
    num_tasks = 3

    context_fn = generate_binary_context

    psp = PSPLinear(
        in_features,
        out_features,
        num_tasks=num_tasks,
        context_fn=context_fn,
        seed=123,
    )

    contexts = []

    for t in range(num_tasks):
        psp.set_task(t)
        ctx = psp.active_context.clone()
        contexts.append(ctx)

    # Verify all tasks have *some* differences — not guaranteed to be all different,
    # but extremely likely unless the generator behaved incorrectly.
    equal_pairs = sum(
        torch.equal(contexts[i], contexts[j])
        for i in range(num_tasks)
        for j in range(i + 1, num_tasks)
    )
    assert equal_pairs == 0  # extremely unlikely unless something is wrong
