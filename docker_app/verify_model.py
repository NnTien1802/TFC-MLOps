"""Fail build/startup when either checkpoint is absent or incompatible."""
from pathlib import Path
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "TFC"))
from config_files.HAR_Configs import Config
from model import TFC, target_classifier
MODEL_FILE = ROOT / "models" / "tfc_har_gesture_model.pt"
CLASSIFIER_FILE = ROOT / "models" / "tfc_har_gesture_classifier.pt"

def load_models():
    config = Config()
    if config.TSlength_aligned != 206:
        raise RuntimeError(f"Expected input length 206, got {config.TSlength_aligned}")
    config.num_classes_target = 8
    model = TFC(config)
    classifier = target_classifier(config)
    model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu", weights_only=True), strict=True)
    classifier.load_state_dict(torch.load(CLASSIFIER_FILE, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    classifier.eval()
    return model, classifier

def infer(model, classifier, samples):
    """Samples must already be preprocessed like HAR->Gesture training data."""
    x_time = torch.as_tensor(samples, dtype=torch.float32).reshape(1, 1, 206)
    x_freq = torch.abs(torch.fft.fft(x_time, dim=-1))
    with torch.inference_mode():
        _, z_time, _, z_freq = model(x_time, x_freq)
        scores = classifier(torch.cat([z_time, z_freq], dim=1))
        probabilities = torch.softmax(scores, dim=1)
    return int(probabilities.argmax(dim=1).item()), probabilities[0].tolist()

def main():
    torch.set_num_threads(2)
    model, classifier = load_models()
    idx, probs = infer(model, classifier, [0.0] * 206)
    if len(probs) != 8 or not (0 <= idx < 8):
        raise RuntimeError("Incorrect Gesture classifier output shape")
    print("Verified TF-C encoder + Gesture classifier: 206 inputs -> 8 output scores")

if __name__ == "__main__":
    main()
