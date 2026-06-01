import pytest
import torch

from mini_flash_attention import forward


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("B", "H", "N", "D"),
    [
        (1, 1, 64, 32),
        (1, 2, 128, 64),
        (2, 1, 256, 64),
        (2, 3, 128, 128),
        (1, 2, 2048, 64),
    ],
)
def test_forward_matches_naive_attention(B, H, N, D):
    torch.manual_seed(0)

    causal = False

    q = torch.randn((B, H, N, D), device="cuda", dtype=torch.float16).contiguous()
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    scale = D ** -0.5
    scores = q.float() @ k.float().transpose(-2, -1) * scale
    if causal:
        mask = torch.ones((N, N), dtype=torch.bool, device=q.device).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    expected = (probs @ v.float()).to(torch.float16)

    actual = forward(q, k, v, causal=causal)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
