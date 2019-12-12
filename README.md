# Final Project at Spiced Academy

This project is still ongoing.

## Landcover classification and semantic segmentation of satellite imagery

As part of the final project for the Data Science Bootcamp at Spiced Academy, the processing of satellite imagery from various sources by means of CNN models is considered.
Two main avenues are being considered:

1. Landcover classification based on a combination of Sentinel-2 RGB(NIR) and Corine Landcover imagery
1. Semantic segmentation of satellite imagery of roof-top/cars/etc based on med-to-high resolution imagery

A specific goal intended is to show a feasibility of deployment of the two approaches for live inference. Possible application areas could be land-surveilance (e.g. illegal deforestation, or waste dumping) or crop/soil monitoring for agricultural purposes.

## Roof-top Segmentation

A roof-top segmentation model was built based on the [AIRS dataset](https://www.airs-dataset.com/) covering 457sq km of the city of Auckland, NZ.

## Car Segmentation

A car segmentation is being built, based on the [2D Semantic Labeling Contest - Potsdam](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-potsdam.html). The intended challenge is to combine the resulting model with the roof-top segmentation model, into a single live infer-able model.
