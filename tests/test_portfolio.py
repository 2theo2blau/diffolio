from __future__ import annotations

import pytest
import torch

from diffolio.portfolio import l1_normalize, risk_sizes, top_k_mask, top_k_normalize


def test_risk_sizes_follow_eq_16_with_the_floor_first():
    # Plan 6.1: floor(224 / 5) = 44, then multiply.
    assert risk_sizes(224, 5) == [220, 176, 132, 88, 44]
    assert risk_sizes(21, 5) == [20, 16, 12, 8, 4]
    assert risk_sizes(5, 5) == [5, 4, 3, 2, 1]

    for n in range(5, 60):
        k = risk_sizes(n, 5)
        assert all(a > b for a, b in zip(k, k[1:]))
        assert k[-1] == n // 5 >= 1
        assert k[0] <= n


def test_risk_sizes_reject_impossible_settings():
    with pytest.raises(ValueError):
        risk_sizes(4, 5)  # floor(4 / 5) = 0 assets at the top risk level
    with pytest.raises(ValueError):
        risk_sizes(10, 1)


def test_l1_normalize_is_eq_12_and_differentiable():
    x = torch.tensor([[0.02, -0.01, 0.01], [-0.3, 0.1, 0.0]], requires_grad=True)
    w = l1_normalize(x)

    torch.testing.assert_close(w.detach()[0], torch.tensor([0.5, -0.25, 0.25]))
    torch.testing.assert_close(w.detach().abs().sum(-1), torch.ones(2))

    # Plan cross-cutting: the gradient must flow through the denominator too,
    # so d(w_0)/dx_1 is non-zero even though x_1 is not in w_0's numerator.
    (grad,) = torch.autograd.grad(w[0, 0], x)
    assert grad[0, 1] != 0
    assert torch.isfinite(grad).all()


def test_l1_normalize_maps_zero_to_zero_not_nan():
    w = l1_normalize(torch.zeros(2, 4))
    assert torch.equal(w, torch.zeros(2, 4))


def test_top_k_normalize_keeps_signed_largest_magnitudes():
    r = torch.tensor([0.01, -0.05, 0.02, 0.00, -0.03])
    w = top_k_normalize(r, 3)
    # |r| ranking: -0.05, -0.03, 0.02; sum of |kept| = 0.10.
    torch.testing.assert_close(w, torch.tensor([0.0, -0.5, 0.2, 0.0, -0.3]))


def test_top_k_broadcasts_one_k_per_leading_row():
    r = torch.tensor([[0.04, -0.01, 0.03, -0.02]])
    k = torch.tensor([4, 2, 1])
    w = top_k_normalize(r.unsqueeze(-2), k)  # (1, 3, 4)

    assert w.shape == (1, 3, 4)
    assert (w != 0).sum(-1).tolist() == [[4, 2, 1]]
    torch.testing.assert_close(w[0, 2], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(w.abs().sum(-1), torch.ones(1, 3))


def test_top_k_never_selects_invalid_entries():
    r = torch.tensor([0.5, 0.01, -0.02, 0.03])
    valid = torch.tensor([False, True, True, True])

    mask = top_k_mask(r, 2, valid)
    assert mask.tolist() == [False, False, True, True]

    # Asking for more than the valid count keeps only the valid ones.
    w = top_k_normalize(r, 4, valid)
    assert w[0] == 0
    torch.testing.assert_close(w.abs().sum(), torch.tensor(1.0))


def test_top_k_ties_break_deterministically_towards_lower_index():
    r = torch.tensor([0.01, -0.01, 0.01, 0.01])
    assert top_k_mask(r, 2).tolist() == [True, True, False, False]
