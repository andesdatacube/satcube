# Quickstart

## El pipeline completo en una llamada

`process_all` encadena las diez etapas, de imágenes crudas a cubo super-resuelto, y deja cada paso en su propia subcarpeta numerada para que puedas auditar el resultado intermedio.

```python
import ee
import satcube

ee.Authenticate()
ee.Initialize(project="ee-your-project")

# Un parche chico corre rapido de inicio a fin.
meta = satcube.metadata(
    lon=-77.06165, lat=-9.53704,
    width=256, height=256,
    start="2018-01-01", end="2019-12-31",
    max_cloud=30,
).select_bands("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12")

crudo = meta.download(output_dir="produccion/raw")

final = crudo.process_all(
    output_dir="produccion",
    min_clear_pct=70,        # calidad minima tras enmascarar nubes
    max_remaining_gaps=1.0,  # huecos maximos tras gap-filling
    device="cuda",           # "cpu" si no tienes GPU
    sr_variant="SEN2SRLite",
)
print(f"{len(final)} composites mensuales super-resueltos a 2.5 m")
```

El resultado final queda en `produccion/08_superresolved`.

## Animación del cubo

```python
from pathlib import Path
import numpy as np, rasterio as rio
from PIL import Image, ImageDraw
import imageio.v2 as imageio
from IPython.display import Image as IPImage

def rgb(path, size=512):
    with rio.open(path) as src:
        b = (4, 3, 2) if src.count >= 13 else (3, 2, 1)
        d = src.read(b).astype("float32") / 10000.0
    im = Image.fromarray((np.clip(d.transpose(1, 2, 0) * 3.0, 0, 1) * 255).astype("uint8")).resize((size, size))
    ImageDraw.Draw(im).text((12, 10), path.stem, fill=(255, 255, 0))
    return np.array(im)

frames = [rgb(p) for p in sorted(Path("produccion/08_superresolved").glob("*.tif"))]
imageio.mimsave("cubo.gif", frames, duration=0.8, loop=0)
IPImage("cubo.gif")
```

## Siguiente paso

Para entender qué hace cada etapa por dentro, revisa la guía del [Pipeline](pipeline.md).
