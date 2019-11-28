"""

"""

################################################################################
# %% IMPORT PACKAGE
################################################################################

import ee

################################################################################
# %% FUNCTIONS
################################################################################

def maskS2clouds(image):
    qa = image.select('QA60')

    ##### Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = ee.Number(2).pow(10).int();
    cirrusBitMask = ee.Number(2).pow(11).int();

    ##### Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudBitMask).eq(0)and(qa.bitwiseAnd(cirrusBitMask).eq(0))

    return image.updateMask(mask).divide(10000)

################################################################################
# %% INIT EARTH ENGINE API
################################################################################

ee.Initialize()

################################################################################
# %% DEFINE BOUNDING BOX AND TIME FOR ROI
################################################################################

#bbox = ee.Geometry.Rectangle(coords=[ 13.95, 52.21,  12.86, 52.75]) ##### BERLIN
##bbox = ee.Geometry.Rectangle(coords=[-8.86, 39.11, -9.51, 38.47]) ##### LISBON
box = ee.Geometry.Rectangle(coords=[-1.04, 45.11, -0.45, 44.72]) ##### BORDEAUX
#bbox = ee.Geometry.Rectangle(coords=[ 54.02, 24.65,  54.97, 23.95]) ##### ABU DHABI
#bbox = ee.Geometry.Rectangle(coords=[-94.13, 41.92, -93.20, 41.33]) ##### DES MOINES
#bbox =  ee.Geometry.Rectangle(coords=[-105.437, 40.127, -105.093, 39.913]) ##### BOULDER
city = 'bordeaux'
year = '2015'
tstart = '2015-06-23'
tend = '2015-10-01'

################################################################################
# %% SET MAP SOURCE AND FILTERS OF IMAGE DATA
################################################################################

"""
Sentinel-2 is a wide-swath, high-resolution, multi-spectral imaging mission
supporting Copernicus Land Monitoring studies, including the monitoring of
vegetation, soil and water cover, as well as observation of inland waterways
and coastal areas.

The Sentinel-2 L2 data are downloaded from COPERNICUS. They were computed by
running sen2cor. The assets contain 12 UINT16 spectral bands representing SR
scaled by 10000 (unlike in L1 data, there is no B10). There are also several
more L2-specific bands (see band list for details). See the Sentinel-2 User
Handbook for details. In addition, three QA bands are present where one (QA60)
is a bitmask band with cloud mask information. For more details, see the full
explanation of how cloud masks are computed.

EE asset ids for Sentinel-2 L2 assets have the following format:
COPERNICUS/S2_SR/20151128T002653_20151128T102149_T56MNN. Here the first numeric
part represents the sensing date and time, the second numeric part represents
the product generation date and time, and the final 6-character string is a
unique granule identifier indicating its UTM grid reference (see MGRS).

For more details on Sentinel-2 radiometric resolution, see this page.
"""

##### ADDRESS DATASET
collection = ee.ImageCollection("COPERNICUS/S2")

##### FILTER BY DATE
collection = collection.filterDate(tstart, tend)

##### FILTER BY LOW CLOUD COVERAGE
collection = collection.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

##### FILTER BY ROI
collection = collection.filterBounds(bbox)

##### CLOUD MASK FILTER
collection = collection.map(maskS2clouds)

##### CREATE IMAGE FROM MEDIAN
image = collection.median()

##### SELECT RGB CHANNELS
image = image.select(['B4', 'B3', 'B2', 'B8'])

##### CONVERT TO 8BIT COLORS
image = image.multiply(512).uint8()

task = ee.batch.Export.image.toDrive(
    image,
    description=f'image-{city}-{year}',
    region=bbox.getInfo()['coordinates'],
    scale=10,
    folder='Landcover-classification',
    #crs='EPSG:4326',
    )

task.start()

################################################################################
# %% SET MAP SOURCE AND FILTERS OF IMAGE DATA
################################################################################

"""
The MCD12Q1 V6 product provides global land cover types at yearly intervals
(2001-2016) derived from six different classification schemes. It is derived
using supervised classifications of MODIS Terra and Aqua reflectance data.
The supervised classifications then undergo additional post-processing that
incorporate prior knowledge and ancillary information to further refine
specific classes.
"""

##### ADDRESS DATASET
#collection = ee.ImageCollection("MODIS/006/MCD12Q1") ##### 500m, but skewed
#collection = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG') ##### 450m, bleed
image = ee.Image('COPERNICUS/CORINE/V18_5_1/100m/2012') ### 100m, Europe only, but nice! Overlayable with folium?
#image = ee.Image('ESA/GLOBCOVER_L4_200901_200912_V2_3');

##### FILTER BY DATE
collection = collection.filterDate(tstart, tend)

##### FILTER BY ROI
collection = collection.filterBounds(bbox)

##### CREATE IMAGE FROM MEDIAN
#image = collection.median()

#collection.getInfo()

##### SELECT RGB CHANNELS
#image = image.select(['LC_Type1'])
#image = image.select(['avg_rad'])

##### CONVERT TO 8BIT COLORS
#image = image.multiply(512).uint8()

task = ee.batch.Export.image.toDrive(
    image,
    description=f'landcover-{city}-{year}',
    region=bbox.getInfo()['coordinates'],
    scale=500,
    folder='Landcover-classification',
    crs='EPSG:4326',
    )

task.start()
