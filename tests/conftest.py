
import json
import os

import pytest


def pytest_addoption(parser):
    parser.addoption("--data-root", action="store", default=os.environ.get("WNW_DATA_ROOT", ""),
                     help="Root directory containing datasets and models")
    parser.addoption("--model-path", action="store", default=None,
                     help="Path to the model to use for tests")
    parser.addoption("--adapter", action="store", default=None,
                     help="Adapter name (voxtral, qwen2_5_omni)")
    parser.addoption("--samples", action="store", default=None,
                     help="Path to sample jsonl file")
    parser.addoption("--device", action="store", default="cuda:0",
                     help="Torch device for model execution")


@pytest.fixture(scope="session")
def data_root(request):
    return request.config.getoption("--data-root")


@pytest.fixture(scope="session")
def device(request):
    return request.config.getoption("--device")


@pytest.fixture(scope="session")
def model_path(request):
    path = request.config.getoption("--model-path")
    if not path:
        pytest.skip("--model-path is required for this test")
    return path


@pytest.fixture(scope="session")
def adapter_name(request):
    return request.config.getoption("--adapter")


@pytest.fixture(scope="session")
def samples_path(request):
    path = request.config.getoption("--samples")
    if not path:
        pytest.skip("--samples is required for this test")
    return path


@pytest.fixture(scope="session")
def first_sample(samples_path):
    with open(samples_path) as f:
        return json.loads(f.readline())


def resolve_audio_path(sample, data_root):
    """Return absolute audio path, joining with data_root if needed."""
    audio_path = sample.get("audio") or sample.get("audio_path")
    if audio_path and not os.path.isabs(audio_path) and data_root:
        audio_path = os.path.join(data_root, audio_path)
    return audio_path
