"""Legend OCR on rapidocr 3.x: bundled models, flags not carried over between calls, example legends."""
import os

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from pyPetrograph.amics import legend_read as lr

EX = os.path.join(os.path.dirname(__file__), "..", "..", "example_data")
FILES = ("ch_PP-OCRv4_det_infer.onnx", "ch_PP-OCRv4_rec_infer.onnx", "ch_ppocr_mobile_v2.0_cls_infer.onnx")


def _text(s: str) -> np.ndarray:
    im = Image.new("RGB", (40 + 22 * len(s), 70), (255, 255, 255))
    ImageDraw.Draw(im).text((20, 15), s, fill=(0, 0, 0), font=ImageFont.load_default(size=36))
    return np.asarray(im)


def test_bundled_models():
    for f in FILES:
        assert os.path.getsize(os.path.join(lr._MODELS, f)) > 100_000


def test_flags_not_carried_over():
    """rapidocr 3.x remembers use_det/use_cls/use_rec: a rec-only call must not break the next ocr_lines."""
    eng = lr._get_engine()
    if eng is None:
        pytest.skip("rapidocr not installed")
    img = _text("Quartz 1500")
    first = [t for _, t, _ in lr.ocr_lines(eng, img)]
    r = eng(img, use_det=False, use_cls=False, use_rec=True)   # e.g. another caller reading one line
    assert getattr(r, "boxes", None) is None   # rec-only result
    again = lr.ocr_lines(eng, img)
    assert [t for _, t, _ in again] == first and "Quartz" in " ".join(first)
    assert all(np.asarray(b).shape == (4, 2) for b, _, _ in again)


@pytest.mark.parametrize("name,n,first", [("10-Mineral_Legend.tiff", 17, "Monazite & Xenotime"),
                                          ("Con-1_Mineral_Legend.tiff", 21, "Sanidine / Orthoclase")])
def test_example_legends(name, n, first):
    if lr._get_engine() is None:
        pytest.skip("rapidocr not installed")
    legend, _ = lr.read_legend(os.path.join(EX, name))
    assert len(legend) == n and next(iter(legend)) == first
    assert not any(k.startswith("swatch_") for k in legend)
