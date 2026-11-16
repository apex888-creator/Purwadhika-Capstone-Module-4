"""YOLO call arguments must not contain the deprecated `half`."""
from purwa_yolo import device


def test_half_is_never_passed():
    assert "half" not in device.INFER_KWARGS


def test_cpu_passes_only_the_device():
    if device.DEVICE == "cpu":
        assert device.INFER_KWARGS == {"device": "cpu"}


def test_gpu_uses_the_new_quantize_argument():
    if device.DEVICE != "cpu":
        assert device.INFER_KWARGS["quantize"] == 16
