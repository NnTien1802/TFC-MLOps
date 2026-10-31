"""TF-C original architecture smoke tests. Does not retrain or modify the paper's model."""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'code' / 'TFC'))
from config_files.HAR_Configs import Config  # noqa: E402
from model import TFC, target_classifier  # noqa: E402


def test_har_config_and_tfc_forward():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    config = Config()
    assert config.TSlength_aligned == 206, 'Unexpected HAR signal length'
    tfc = TFC(config).eval()
    x_t = torch.randn(4, 1, config.TSlength_aligned)
    x_f = torch.abs(torch.fft.fft(x_t, dim=-1))
    with torch.no_grad():
        h_t, z_t, h_f, z_f = tfc(x_t, x_f)
    assert (h_t.shape, h_f.shape) == ((4, 206), (4, 206))
    assert (z_t.shape, z_f.shape) == ((4, 128), (4, 128))
    assert all(torch.isfinite(x).all() for x in (h_t, z_t, h_f, z_f))


def test_finetune_classifier_and_backward():
    torch.manual_seed(123)
    torch.set_num_threads(2)
    config = Config()
    tfc = TFC(config).train()
    classifier = target_classifier(config).train()
    x_t = torch.randn(4, 1, config.TSlength_aligned)
    x_f = torch.abs(torch.fft.fft(x_t, dim=-1))
    _, z_t, _, z_f = tfc(x_t, x_f)
    logits = classifier(torch.cat([z_t, z_f], dim=1))
    assert logits.shape == (4, config.num_classes_target)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0, 1]))
    loss.backward()
    assert torch.isfinite(loss)
    assert tfc.projector_t[0].weight.grad is not None
    assert tfc.projector_f[0].weight.grad is not None
    assert classifier.logits_simple.weight.grad is not None
