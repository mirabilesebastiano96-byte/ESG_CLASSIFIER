import pytest

torch = pytest.importorskip("torch")

from esgpillar.models import DNN, BiGRU, BiLSTM, ConvNet  # noqa: E402


@pytest.mark.parametrize(
    "cls,is_seq", [(DNN, False), (ConvNet, True), (BiLSTM, True), (BiGRU, True)]
)
def test_forward_pass_shapes(cls, is_seq):
    d, batch, seqlen = 16, 4, 10
    model = cls(d=d)
    if is_seq:
        x = torch.randn(batch, seqlen, d)
        lengths = torch.full((batch,), seqlen, dtype=torch.long)
        out = model(x, lengths)
    else:
        x = torch.randn(batch, d)
        out = model(x)
    assert out.shape == (batch, 3)
