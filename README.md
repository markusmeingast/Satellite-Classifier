[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Final Project at Spiced Academy

The associated presentation can be found [here](https://markusmeingast.github.io/Satellite-Classifier/)

## Semantic Segmentation of Satellite Imagery

As part of the final project for the Data Science Bootcamp at Spiced Academy,
the processing of satellite imagery from various sources by means of CNN models
is considered. Two main topics are being considered:

1. Semantic segmentation of satellite imagery of roof-top/roads/etc based on med-to-high resolution aerial or satellite imagery
1. Fusing multiple segmentation models based on different datasets into a single edge-deployable model for inference.

Possible application areas could be land-surveillance (e.g. illegal deforestation, or waste dumping), crop/soil monitoring for agricultural purposes or disaster relief coordination.

## Roof-top Segmentation

A roof-top segmentation model was built based on the [AIRS dataset](https://www.airs-dataset.com/) covering 457sq km of the city of Auckland, NZ.

## Car Segmentation

A car segmentation is was built, based on the [2D Semantic Labeling Contest - Potsdam](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-potsdam.html).

## Roads segmentation

A road segmentation model was built, using the [RoadNet](https://github.com/yhlleo/RoadNet) dataset. The intended challenge is to combine the resulting models with the roof-top segmentation model, into a single live infer-able model.

## Documentation

The associated presentation can be found [here](https://markusmeingast.github.io/Satellite-Classifier/)

# References

- Chen, Qi, Lei Wang, Yifan Wu, Guangming Wu, Zhiling Guo, and Steven L. Waslander. "Aerial imagery for roof segmentation: A large-scale dataset towards automatic mapping of buildings." arXiv preprint arXiv:1807.09532 (2018).

- Liu, Yahui, Jian Yao, Xiaohu Lu, Menghan Xia, Xingbo Wang, and Yuan Liu. "RoadNet: Learning to Comprehensively Analyze Road Networks in Complex Urban Scenes From High-Resolution Remotely Sensed Images." IEEE Transactions on Geoscience and Remote Sensing 57, no. 4 (2018): 2043-2056.

- Rottensteiner, Franz, Gunho Sohn, Markus Gerke and Jan Dirk Wegner. "ISPRS Test Project on Urban Classification and 3D Building Reconstruction" ISPRS - Commission III (2013)
