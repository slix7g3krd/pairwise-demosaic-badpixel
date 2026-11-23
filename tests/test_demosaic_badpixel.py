import pytest
import imaging.demosaic_badpixel as mod

CASES = [
    ("bad_pixel", (1)),
    ("phase_after_crop", (1)),
    ("order_by_iso", (1)),
    ("parallel_reduce", (1)),
    ("false_color", (1)),
    ("exif_orient", (1)),
    ("denoise_order", (1)),
    ("validation", (1)),
    ("audit", (1)),
    ("fallback", (1)),
    ("raw_meta", (1)),
    ("metrics", (1)),
    ("edge_guard", (1)),
    ("batch_repeat", (1)),
    ("determinism", (1)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
