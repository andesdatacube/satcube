# Instalación

satcube se instala desde PyPI:

```bash
pip install satcube
```

Esto trae también las dependencias clave del pipeline (cubexpress para el acceso, y las piezas de nubes, alineación y super-resolución).

## Requisitos

- Python 3.10 o superior.
- Una cuenta de Google Earth Engine para descargar Sentinel-2. La primera vez autentícate:

```python
import ee
ee.Authenticate()
ee.Initialize(project="ee-your-project")   # reemplaza por tu project id
```

## GPU (recomendado)

Las etapas de deep learning (enmascarado de nubes y super-resolución) corren en CPU, pero son mucho más rápidas en GPU. Si tienes una, pasa `device="cuda"` a `cloud_masking`, `superresolve` o `process_all`. En CPU usa `device="cpu"`.

## Verifica la instalación

```python
import satcube
print(satcube.__version__)
```
