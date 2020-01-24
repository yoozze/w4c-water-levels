# Surface water modelling
Water4Cities project...

## Software requirements

### Dockerized version

```
docker run -p 8888:8888 -v "$PWD":/home/jovyan jupyter/datascience-notebook
# Check the right Docker instance id with "docker ps"
docker exec b7f3abbf54da pip install ipywidgets scikit-multiflow gmaps asyncio aiohttp
```


### Installing on machine
* [Anaconda (python 3.7 or higher)](http://conda.io/docs/user-guide/install/index.html)

## Installation
Clone repository:
```
git clone https://github.com/yoozze/w4c-stream.git
```

Run installation script (Windows only):
```
cd w4c-stream
install.bat
```
Or install manually:
```
conda create -n w4c
conda activate w4c
conda install nb_conda
pip install ipywidgets scikit-multiflow gmaps asyncio aiohttp
jupyter nbextension enable --py gmaps
```
Start Jupyter notebook:
```
jupyter notebook
```
For google maps to work correctly, you need to set your google API key to `GOOGLE_API_KEY` environment variable:
```
setx GOOGLE_API_KEY your_api_key
```

## Data acquisition
Build database:
```
cd data
python get_water.py
python get_weather.py
python build_db.py
```
