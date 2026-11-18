import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from main import gradient, descend, load_dem


def test_plane_anisotropic_pixels_and_direction():
    r, c = np.mgrid[:51, :61]
    z = 100 + 0.2 * c * 3 + 0.1 * r * 7
    gx, gy, slope = gradient(z, 3, 7)
    np.testing.assert_allclose(gx, 0.2, atol=1e-12)
    np.testing.assert_allclose(gy, -0.1, atol=1e-12)
    np.testing.assert_allclose(slope, np.degrees(np.arctan(np.hypot(.2, .1))))
    t = descend(z, 3, 7, (25, 30), max_steps=10)
    assert np.all(np.diff(t.heights) < 0)
    dr, dc = t.rc[-1] - t.rc[0]
    assert dr < 0 and dc < 0
    assert (dr * 7) / (dc * 3) == pytest.approx(.5)


def test_bowl_converges():
    r, c = np.mgrid[:101, :101]
    z = ((r - 50.) ** 2 + (c - 50.) ** 2)
    t = descend(z, 1, 1, (15, 20))
    assert np.linalg.norm(t.rc[-1] - [50, 50]) < 1
    assert np.all(np.diff(t.heights) < 0)


def test_flat_and_nodata():
    z = np.ones((20, 20))
    assert descend(z, 1, 1, (10, 10)).reason == "stationary"
    z[10, 10] = np.nan
    with pytest.raises(ValueError):
        descend(z, 1, 1, (10, 10))
    assert np.isnan(gradient(z, 1, 1)[0][10, 10])


@pytest.mark.parametrize("crs,transform", [
    ("EPSG:32637", from_origin(500000, 5000000, 10, 20)),
    ("EPSG:4326", from_origin(38, 45, .001, .001)),
])
def test_geotiff(tmp_path, crs, transform):
    path = tmp_path / "test.tif"
    z = np.full((20, 20), 123, dtype="float32")
    z[0, 0] = -9999
    with rasterio.open(path, "w", driver="GTiff", width=20, height=20, count=1,
                       dtype="float32", crs=crs, transform=transform, nodata=-9999) as dst:
        dst.write(z, 1)
    dem = load_dem(path)
    assert dem.crs.is_projected
    assert dem.dx > 0 and dem.dy > 0
    np.testing.assert_allclose(dem.z[np.isfinite(dem.z)], 123)
    assert np.isnan(dem.z).any()
