import numpy as np
import OpenEXR
import pytest

from usd_synth.passes import read_passes


def test_read_passes_rejects_fractional_large_ids(tmp_path):
    index = np.full((4, 4), 1000.0, dtype=np.float32)
    index[0, 0] = 1000.005  # beyond the 1e-3 tolerance, within a relative one
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    channels = {
        "ViewLayer.Depth.Z": np.ones((4, 4), dtype=np.float32),
        "ViewLayer.Object Index.X": index,
    }
    OpenEXR.File(header, channels).write(str(tmp_path / "aux.exr"))
    with pytest.raises(ValueError, match="not integral"):
        read_passes(tmp_path / "aux.exr")


def test_read_passes_rounds_exact_ids(tmp_path):
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    index = np.array([[0, 1], [2, 1000]], dtype=np.float32)
    channels = {"ViewLayer.Depth.Z": np.ones((2, 2), np.float32), "ViewLayer.Object Index.X": index}
    OpenEXR.File(header, channels).write(str(tmp_path / "aux.exr"))
    p = read_passes(tmp_path / "aux.exr")
    assert p.index.tolist() == [[0, 1], [2, 1000]] and p.index.dtype == np.int32
