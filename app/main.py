"""A lightweight deployment-status web service for the original TF-C repository.

This does not claim to perform HAR/Gesture inference. It intentionally keeps
training/inference implementations in code/TFC/ unchanged.
"""
from pathlib import Path
from fastapi import FastAPI

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title='TF-C Deployment Status', version='1.0')


@app.get('/')
def root():
    return {
        'application': 'Original TF-C',
        'ci_cd': 'GitHub Actions + Render',
        'model': 'TF-C Time-Frequency Consistency',
        'dataset_protocol': 'HAR -> Gesture',
        'inference_available': False,
        'message': 'Deployment status API only; no Gesture classifier inference provided.',
    }


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.get('/model-info')
def model_info():
    return {
        'model_source_present': (ROOT / 'code' / 'TFC' / 'model.py').is_file(),
        'checkpoint_in_repo': (ROOT / 'ckp_last.pt').is_file(),
        'status': 'CI/CD deployment demonstration; inference not enabled',
    }
