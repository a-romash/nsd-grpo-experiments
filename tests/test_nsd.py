import torch

from src.nsd import nsd_loss, select_nsd_rows


def test_routing_zero():
    rewards = torch.tensor([0., 0., 0., 0., 0., 1., 0., 0., 1., 1., 1., 1.])
    selected = select_nsd_rows(rewards, 4, "zero")
    assert selected.tolist() == [True] * 4 + [False] * 8
    assert select_nsd_rows(rewards, 4, "all").sum() == 12
    assert select_nsd_rows(rewards, 4, "off").sum() == 0


def test_gating_and_gradients():
    student = torch.tensor([[-1.6, -0.3], [-0.2, -1.1]], requires_grad=True)
    reference = torch.log(torch.tensor([[0.2, 0.4], [0.5, 0.3]]))
    negative = torch.log(torch.tensor([[0.4, 0.1], [0.7, 0.2]]))
    mask = torch.ones_like(student)
    selected = torch.tensor([True, False])
    loss, stats = nsd_loss(student, reference, negative, mask, selected, alpha=0.01)
    assert torch.isfinite(loss)
    loss.backward()
    assert student.grad[0].abs().sum() > 0
    assert student.grad[1].abs().sum() == 0
    assert abs(stats["nsd/active_rows"].item() - 0.5) < 1e-6


def test_no_selected_loss_is_zero():
    student = torch.tensor([[-1.2]], requires_grad=True)
    ref = torch.tensor([[-1.1]])
    neg = torch.tensor([[-0.7]])
    loss, _ = nsd_loss(student, ref, neg, torch.ones_like(student), torch.tensor([False]), 0.01)
    assert loss.item() == 0
