"""Рельеф: загрузка GeoTIFF, градиент, уклоны и линии спуска."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rasterio.transform import from_origin

from dataclasses import dataclass

import numpy as np
import rasterio
from pyproj import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
from scipy.ndimage import map_coordinates


@dataclass
class DEM:
    z: np.ndarray
    transform: rasterio.Affine
    crs: CRS

    @property
    def dx(self):
        return self.transform.a

    @property
    def dy(self):
        return -self.transform.e


def load_dem(path, target_crs=None, max_cells=4_000_000):
    """Загрузить матрицу высот и привести сетку к координатам в метрах.

    Верх изображения соответствует северу. Высоты в файле должны быть в метрах.
    """
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError("GeoTIFF не содержит CRS.")
        if src.width * src.height > max_cells:
            raise ValueError("Слишком большой DEM: сначала вырежьте небольшой участок.")
        crs = CRS(src.crs)
        metric = crs.is_projected and all(
            abs(a.unit_conversion_factor - 1) < 1e-9 for a in crs.axis_info[:2]
        )
        if target_crs is not None:
            dst_crs = CRS(target_crs)
        elif metric:
            dst_crs = crs
        else:
            west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
            lon, lat = (west + east) / 2, (south + north) / 2
            if not (-80 <= lat <= 84) or east - west > 6 or north - south > 6:
                raise ValueError("Для этого участка задайте подходящую метрическую target_crs.")
            zone = min(60, max(1, int((lon + 180) // 6) + 1))
            dst_crs = CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)
        if not dst_crs.is_projected or any(
            abs(a.unit_conversion_factor - 1) > 1e-9 for a in dst_crs.axis_info[:2]
        ):
            raise ValueError("Целевая CRS должна иметь горизонтальные единицы в метрах.")
        z = src.read(1, masked=True).astype(float).filled(np.nan)
        t = src.transform
        if dst_crs != crs or t.b != 0 or t.d != 0 or t.a <= 0 or t.e >= 0:
            t, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )
            if width * height > max_cells:
                raise ValueError("Результат перепроецирования слишком большой.")
            dest = np.full((height, width), np.nan)
            reproject(z, dest, src_transform=src.transform, src_crs=src.crs,
                      dst_transform=t, dst_crs=dst_crs, src_nodata=np.nan,
                      dst_nodata=np.nan, resampling=Resampling.bilinear)
            z = dest
        if min(z.shape) < 3 or not np.isfinite(z).any():
            raise ValueError("DEM должен содержать высоты и хотя бы 3×3 ячейки.")
        return DEM(z, t, dst_crs)


def gradient(z, dx, dy):
    """Вычислить производные на восток, на север и уклон в градусах.

    Номера строк растут на юг, поэтому знак производной по строкам меняется.
    Пропуски высот делают соседние производные неопределёнными.
    """
    z = np.asarray(z, dtype=float)
    if z.ndim != 2 or min(z.shape) < 3 or not (dx > 0 and dy > 0):
        raise ValueError("Нужна матрица минимум 3×3 и положительные размеры пикселя.")
    south, east = np.gradient(z, dy, dx, edge_order=2)
    north = -south
    east[~np.isfinite(z)] = np.nan
    north[~np.isfinite(z)] = np.nan
    return east, north, np.degrees(np.arctan(np.hypot(east, north)))


@dataclass
class Trace:
    rc: np.ndarray
    heights: np.ndarray
    reason: str


def descend(z, dx, dy, start, step=None, max_steps=5000, tolerance=1e-8):
    """Построить линию спуска по отрицательному градиенту.

    Начало start задаётся парой (строка, столбец), шаг step — в метрах.
    Между ячейками используется билинейная интерполяция.
    Если высота не уменьшается, уменьшаем шаг. Скорость воды не рассчитывается.
    """
    gx, gy, _ = gradient(z, dx, dy)
    step = min(dx, dy) * 0.4 if step is None else float(step)
    if not np.isfinite(step) or step <= 0 or max_steps < 1 or tolerance <= 0:
        raise ValueError("Некорректные параметры интегрирования.")
    p = np.asarray(start, dtype=float)
    if p.shape != (2,) or not np.isfinite(p).all():
        raise ValueError("Начало должно содержать два конечных числа: строку и столбец.")

    def inside(q):
        return 0 <= q[0] <= z.shape[0] - 1 and 0 <= q[1] <= z.shape[1] - 1

    def sample(a, q):
        return float(map_coordinates(a, q[:, None], order=1, mode="constant",
                                     cval=np.nan, prefilter=False)[0])

    if not inside(p) or not np.isfinite(sample(z, p)):
        raise ValueError("Начальная точка вне DEM или попала в NoData.")
    points, heights = [p.copy()], [sample(z, p)]
    reason = "max_steps"
    for _ in range(max_steps):
        east, north = sample(gx, p), sample(gy, p)
        norm = np.hypot(east, north)
        if not np.isfinite(norm):
            reason = "nodata"
            break
        if norm <= tolerance:
            reason = "stationary"
            break
        direction = np.array([north / dy, -east / dx]) / norm
        length = step
        accepted = False
        for _ in range(30):
            q = p + length * direction
            if not inside(q):
                reason = "boundary"
            else:
                h = sample(z, q)
                if not np.isfinite(h):
                    reason = "nodata"
                elif h < heights[-1] - tolerance:
                    accepted = True
                    break
                else:
                    reason = "no_descent"
            length *= 0.5
        if not accepted:
            break
        p = q
        points.append(p.copy())
        heights.append(h)
    else:
        reason = "max_steps"
    return Trace(np.array(points), np.array(heights), reason)


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser._positionals.title = "Позиционные аргументы"
    parser._optionals.title = "Параметры"
    parser.add_argument("-h", "--help", action="help", help="показать справку и выйти")
    parser.add_argument("--dem", type=Path, metavar="ФАЙЛ", help="GeoTIFF с высотами; без файла создаётся тестовая чаша")
    parser.add_argument("--out", type=Path, default=Path("outputs"), metavar="ПАПКА", help="папка результатов (по умолчанию outputs)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.dem
    if source is None:
        # Создаём искусственную чашу: минимум в центре, размер ячейки 10×10 м.
        # Её точная формула известна, поэтому на ней удобно проверять спуск.
        source = args.out / "synthetic_bowl.tif"
        r, c = np.mgrid[:101, :101]
        z = 100 + ((c - 50) * 10) ** 2 / 2000 + ((r - 50) * 10) ** 2 / 2000
        with rasterio.open(source, "w", driver="GTiff", width=101, height=101,
                           count=1, dtype="float64", crs="EPSG:32637",
                           transform=from_origin(500000, 5000000, 10, 10)) as dst:
            dst.write(z, 1)
    dem = load_dem(source)
    gx, gy, slope = gradient(dem.z, dem.dx, dem.dy)
    valid = np.argwhere(np.isfinite(gx) & np.isfinite(gy))
    if len(valid) == 0:
        raise ValueError("Нет точек с определённым градиентом.")
    starts = valid[np.linspace(0, len(valid) - 1, 9, dtype=int)]
    traces = [descend(dem.z, dem.dx, dem.dy, p) for p in starts]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, values, title in zip(axes, [dem.z, slope], ["Высота, м", "Уклон, градусы"]):
        im = ax.imshow(values, cmap="terrain" if ax is axes[0] else "magma")
        fig.colorbar(im, ax=ax)
        ax.set(title=title, xlabel="Столбец", ylabel="Строка", aspect=dem.dy / dem.dx)
    for trace in traces:
        axes[0].plot(trace.rc[:, 1], trace.rc[:, 0], color="dodgerblue", linewidth=1.5)
        axes[0].plot(*trace.rc[0, ::-1], "wo", markersize=3)
    fig.savefig(args.out / "terrain.png", dpi=160)
    plt.close(fig)
    print(f"Готово! Рисунок: {args.out.resolve() / 'terrain.png'}")


if __name__ == "__main__":
    main()
