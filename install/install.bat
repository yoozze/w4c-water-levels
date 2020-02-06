@echo off

echo.
echo ******************************
echo * Createing new environment: *
echo ******************************
echo.
call conda create -n w4c
call conda activate w4c

echo.
echo *******************************************
echo * Installing jupyter notebook extensions: *
echo *******************************************
echo.
call conda install nb_conda

echo.
echo ******************************
echo * Installing other packages: *
echo ******************************
echo.
call pip install ipywidgets numpy scikit-multiflow gmaps asyncio aiohttp deap

echo.
echo *****************************
echo * Enabling gmaps extension: *
echo *****************************
echo.
call jupyter nbextension enable --py gmaps

echo.
echo ******************************
echo * Starting Jupyter notebook: *
echo ******************************
echo.
call jupyter notebook

