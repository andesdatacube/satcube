# El pipeline por dentro

satcube encadena diez etapas. Cada método devuelve un nuevo objeto, así que puedes encadenarlos o pararte en cualquier punto. Aquí qué hace cada uno y por qué.

## 1. Metadata y descarga

```python
meta = satcube.metadata(lon, lat, width, height, start=..., end=..., max_cloud=30)
meta = meta.select_bands("B1", ..., "B12")
cubo = meta.download(output_dir="raw")
```

`metadata` descubre las escenas Sentinel-2 sobre tu parche y las puntúa por nubosidad con CloudScore+ (de Google), de modo que `max_cloud` filtra **antes** de descargar. `select_bands` te deja bajar solo las bandas que usarás. `download` guarda un GeoTIFF por fecha.

Para un polígono en vez de un punto, usa `metadata_polygon(geojson, ...)`.

## 2. Co-registro sub-píxel (`align`)

```python
alineado = cubo.align(output_dir="aligned")
```

Imágenes de fechas distintas caen con desfases de fracciones de píxel. `align` elige la escena más limpia como referencia y co-registra el resto contra ella (Phase Cross-Correlation por defecto). Devuelve `dx_px` y `dy_px` por escena para que audites cuánto se movió cada una. Sin esto, los composites mezclan píxeles vecinos y la detección de cambios marca diferencias que no existen.

## 3. Enmascarado de nubes (`cloud_masking`)

```python
masked = alineado.cloud_masking(output_dir="masked", device="cuda", save_mask=True)
limpio = masked[masked["clear_pct"] >= 70]
```

Aplica CloudSEN12 (deep learning) y clasifica cada píxel en despejado, nube delgada, sombra o nube gruesa. A diferencia del puntaje por escena, esto es una **máscara píxel a píxel**: sabe exactamente qué zonas descartar. Devuelve `clear_pct` por escena para que filtres las demasiado nubladas.

## 4. Gap-filling (`gapfill`)

```python
relleno    = limpio.gapfill(output_dir="gapfilled")
relleno_ok = relleno[relleno["remaining_gaps_pct"] <= 1.0]
```

Rellena los huecos que dejaron las nubes trayendo información real de otras fechas. Por cada hueco evalúa hasta diez fechas vecinas y se queda con la de menor error de color, transfiere color con histogram matching y mezcla el borde con un degradado de distancia. Corre en tres rondas: estricta, relajada y un inpainting final para lo que ninguna fecha cubrió.

Devuelve `remaining_gaps_pct`, que se mide **al terminar la ronda estricta**: es el porcentaje de píxeles que ningún match temporal de alta confianza pudo rellenar. Es un medidor de qué tan reconstruible era la escena con datos reales. Filtrar con `<= 1.0` deja solo las escenas reconstruidas casi enteras a partir de fechas vecinas, no del inpainting.

## 5. Composites mensuales (`composite`)

```python
mensual = relleno_ok.composite(output_dir="monthly", agg_method="median")
```

Agrupa las escenas de cada mes y toma la **mediana** píxel a píxel. La mediana es robusta a atípicos: una nube no detectada deja un píxel anormalmente brillante en una sola fecha, y la mediana lo descarta mientras el promedio lo arrastraría. En la práctica es un segundo filtro de nubes muy potente.

## 6. Interpolación (`interpolate`)

```python
interp = mensual.interpolate(output_dir="interp", despike_threshold=0.15)
```

Detecta saltos bruscos en la serie de cada píxel contra su mediana móvil de tres meses (nubes que se colaron), los marca y rellena por interpolación lineal en el tiempo. Sube `despike_threshold` si tienes eventos reales abruptos (incendios, inundaciones) que no quieres borrar.

## 7. Suavizado (`smooth`)

```python
suavizado = interp.smooth(output_dir="smooth", smooth_w=7, smooth_p=2)
```

Savitzky-Golay sobre la anomalía: resta la climatología mensual, suaviza, y la vuelve a sumar, para no aplanar el ciclo estacional. Con un solo año es casi identidad; importa cuando tienes varios años y quieres quitar ruido inter-anual.

## 8. Super-resolución (`superresolve`)

```python
sr = suavizado.superresolve(output_dir="sr", variant="SEN2SRLite", device="cuda")
```

Lleva la serie de 10 m a 2.5 m (aumento 4×) con SEN2SR. Toma las bandas que el modelo necesita y escribe la salida a alta resolución. `SEN2SRLite` es rápido y recomendado; `SEN2SR` es más pesado y de mejor calidad.

## Todo junto

Las ocho etapas (más re-alineación y filtros de calidad) están envueltas en `process_all`. Revisa el [Quickstart](quickstart.md).
