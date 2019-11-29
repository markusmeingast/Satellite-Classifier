"""
Data request script for landcover classification project. This script requests
satellite imagery of theCOPERNICUS/S2 program as well as landcover definitions
of COPERNICUS/CORINE/V18_5_1/100m/2012 program through the Google Earth Engine
API. Various alternative sources are possible through the API, but S2 imagery
and CORINE classification data show most promising starting point.

TODO:

1. CLEAN UP CODE
2. GET EXPORT BY BBOX (MAYBE GEOJSON FILE?) AND DATE/DELTA (MAYBE LIST AVAILABLE TIMEFRAMES FROM DATASET?)
3. FILE NAMES BASED ON DATASET INFO (FROM 2.)
"""

################################################################################
# %% IMPORT PACKAGE
################################################################################

import ee

################################################################################
# %% FUNCTIONS
################################################################################

def maskS2clouds(image):
    """
    Function to remove clouds through masking of the QA60 layer within the
    S2 dataset
    """
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
bbox = ee.Geometry.Rectangle(coords=[-1.04, 45.11, -0.45, 44.72]) ##### BORDEAUX
#bbox = ee.Geometry.Rectangle(coords=[ 54.02, 24.65,  54.97, 23.95]) ##### ABU DHABI
#bbox = ee.Geometry.Rectangle(coords=[-94.13, 41.92, -93.20, 41.33]) ##### DES MOINES
#bbox =  ee.Geometry.Rectangle(coords=[-105.437, 40.127, -105.093, 39.913]) ##### BOULDER
#bbox =  ee.Geometry.Rectangle(coords=[ 10.224, 60.134, 11.350, 59.530]) ##### OSLO
#bbox =  ee.Geometry.Rectangle(coords=[ -4.064, 40.668, -3.356, 40.166]) ##### MADRID

city = 'bordeaux'

year = '2016'
tstart = '2016-06-23'
tend = '2016-10-01'

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

##### SELECT RGB (+NEAR INFRA RED) CHANNELS
image = image.select(['B4', 'B3', 'B2', 'B8'])

##### CONVERT TO 8BIT COLORS
image = image.multiply(512).uint8()

task = ee.batch.Export.image.toDrive(
    image,
    description=f'image-{city}-{year}',
    region=bbox.getInfo()['coordinates'],
    scale=10,
    folder='Landcover-classification',
    crs='EPSG:4326',
    )

task.start()

################################################################################
# %% SET MAP SOURCE AND FILTERS OF IMAGE DATA
################################################################################

"""
The CORINE (coordination of information on the environment) Land Cover (CLC)
inventory was initiated in 1985 to standardize data collection on land in Europe
to support environmental policy development. The project is coordinated by the
European Environment Agency (EEA) in the frame of the EU Copernicus programme
and implemented by national teams. The number of participating countries has
increased over time currently including 33 (EEA) member countries and six
cooperating countries (EEA39) with a total area of over 5.8 Mkm2.

The reference year of the first CLC inventory was 1990 and the first update
created in 2000. Later, the update cycle has become 6 years. Satellite imagery
provides the geometrical and thematic basis for mapping with in-situ data as
essential ancillary information. The basic technical parameters of CLC (i.e. 44
classes in nomenclature, 25 hectares minimum mapping unit (MMU), and 100 meters
minimum mapping width) have not changed since the beginning, therefore the
results of the different inventories are comparable.
"""

##### ADDRESS DATASET
image = ee.Image('COPERNICUS/CORINE/V18_5_1/100m/2012') ### 100m, Europe only, but nice! Overlayable with folium?

##### POSSIBLE ALTERNATIVES
#image = ee.Image('ESA/GLOBCOVER_L4_200901_200912_V2_3');
#collection = ee.ImageCollection("MODIS/006/MCD12Q1") ##### 500m, but skewed
#collection = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG') ##### 450m, bleed

""" RELEVANT FOR COLLECTIONS
##### FILTER BY DATE
#collection = collection.filterDate(tstart, tend)

##### FILTER BY ROI
#collection = collection.filterBounds(bbox)

##### CREATE IMAGE FROM MEDIAN
#image = collection.median()

#collection.getInfo()

##### SELECT RGB CHANNELS
#image = image.select(['LC_Type1'])
#image = image.select(['avg_rad'])

##### CONVERT TO 8BIT COLORS
#image = image.multiply(512).uint8()
"""

task = ee.batch.Export.image.toDrive(
    image,
    description=f'landcover-{city}-2012',
    region=bbox.getInfo()['coordinates'],
    scale=500,
    folder='Landcover-classification',
    crs='EPSG:4326',
    )

task.start()
