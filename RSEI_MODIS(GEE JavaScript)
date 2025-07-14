//Add user-defined ROI named table

Map.addLayer(table, {color:"000000"},"table");
Map.centerObject(table, 6); 

// remove cloud
function maskclouds(image) {
  var qa = image.select('state_1km');
  // remove cloud
  var cloudBitMask = 1 << 10; 
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
  // remove cloud shadow
  var shadowBitMask = 1 << 2;
  var shadowMask = qa.bitwiseAnd(shadowBitMask).eq(0);
  mask = mask.and(shadowMask);
  // remove Pixel is adacent to cloud
  var adjacentBitMask = 1 << 13;
  var adjacentMask = qa.bitwiseAnd(adjacentBitMask).eq(0);
  mask = mask.and(adjacentMask);
  // remove snow
  var snowBitMask = 1 << 15;
  var snowMask = qa.bitwiseAnd(snowBitMask).eq(0);
  mask = mask.and(snowMask);
  
  var surBands = image.select('sur_refl_b.*').multiply(0.0001);
  return image.addBands(surBands, null, true)
  return image.updateMask(mask);
}


//LST
var dataset_lst = ee.ImageCollection('MODIS/061/MOD11A2')
    .filterBounds(table)
    .filterDate('2020-04-24', '2020-10-27')
    .mean();

var mean_lst = dataset_lst.expression(
    'a*0.02-273.15',
    {
        a:dataset_lst.select('LST_Day_1km'), 
    });                   

//NDVI
var dataset_ndvi = ee.ImageCollection('MODIS/061/MOD13A1')
    .filterBounds(table)
    .filterDate('2020-04-24', '2020-10-27')
    .mean();
var mean_ndvi = dataset_ndvi.expression(
    '0.0001*a',
    {
        a:dataset_ndvi.select('NDVI'), 
    });

//remove water by using MOD44W water_musk data 
var datasetwater = ee.ImageCollection('MODIS/006/MOD44W')
    .select('water_mask')
    .filterBounds(table)
    // .map(NDWI_cal)
    .median()
    .clip(table);
   
var datasetwater_mask2 = datasetwater.select('water_mask').lte(0);
var dataset_no_water = datasetwater_mask2.mask(datasetwater_mask2);
print(dataset_no_water,'dataset_no_water');
Map.addLayer(dataset_no_water.clip(table),{},'dataset_no_water');

var dataset = ee.ImageCollection('MODIS/061/MOD09A1')
    .filterDate('2020-04-24', '2020-10-27')
    .filterBounds(table)
    .map(maskclouds)
    .mean()
    .clip(table);
// var dataset = coef(dataset1)
 

//NDBSI
function NDBSI(img){
  var B1 = img.select('sur_refl_b01');
  var B2 = img.select('sur_refl_b02');
  var B3 = img.select('sur_refl_b03');
  var B4 = img.select('sur_refl_b04');
  var B6 = img.select('sur_refl_b06');
  var B7 = img.select('sur_refl_b07');
  // var ndbsi_temp =(((B6.add(B1)).subtract(B2.add(B3))).divide((B6.add(B1)).add(B2.add(B3)))
  //                 .add(B6.multiply(2.0).divide(B6.add(B2)).subtract(B2.divide(B2.add(B1)).add(B4.divide(B4.add(B6))))
  //                 .divide(B6.multiply(2.0).divide(B6.add(B2)).add(B2.divide(B2.add(B1)).add(B4.divide(B4.add(B6))))).divide(2.0)));
    var ndbsi_temp =((((B6.add(B1)).subtract(B2.add(B3))).divide((B6.add(B1)).add(B2.add(B3))))
                  .add(B6.multiply(2.0).divide(B6.add(B2)).subtract(B2.divide(B2.add(B1)).add(B4.divide(B4.add(B6))))
                  .divide(B6.multiply(2.0).divide(B6.add(B2)).add(B2.divide(B2.add(B1)).add(B4.divide(B4.add(B6))))))).divide(2.0);
  return ndbsi_temp;
}
var ndbsi = NDBSI(dataset).mask(dataset_no_water);  


//WET
function WET(img){
  var red = img.select('sur_refl_b01');
  var nir = img.select('sur_refl_b02');
  var blue = img.select('sur_refl_b03');
  var green = img.select('sur_refl_b04');
  var tinf = img.select('sur_refl_b05');
  var bt = img.select('sur_refl_b06');
  var swir = img.select('sur_refl_b07');
  var wet_temp = ((red.multiply(0.1147)).add(nir.multiply(0.2489)).add(blue.multiply(0.2408)).add(green.multiply(0.3132)).subtract(tinf.multiply(0.3122)).subtract(bt.multiply(0.6416)).subtract(swir.multiply(0.5087)));
  return wet_temp;
}
var wet = WET(dataset).mask(dataset_no_water);


function img_normalize(img){
      var minMax = img.reduceRegion({
            reducer:ee.Reducer.minMax(),
            geometry: table,
            scale: 1000,
            maxPixels: 1e13,
        });
      var year = img.get('year');
      var normalize  = ee.ImageCollection.fromImages(
            img.bandNames().map(function(name){
                  name = ee.String(name);
                  var band = img.select(name);
                  return band.unitScale(ee.Number(minMax.get(name.cat('_min'))), ee.Number(minMax.get(name.cat('_max'))));
                    
              })
        ).toBands().rename(img.bandNames());
        return normalize;
}

var unit_ndvi = img_normalize(mean_ndvi);
dataset_no_water=dataset_no_water.addBands(unit_ndvi.rename('NDVI').toFloat());
var unit_NDBSI = img_normalize(ndbsi);
dataset_no_water=dataset_no_water.addBands(unit_NDBSI.rename('NDBSI').toFloat());
var unit_Wet = img_normalize(wet);
dataset_no_water=dataset_no_water.addBands(unit_Wet.rename('Wet').toFloat());
var unit_lst = img_normalize(mean_lst);
dataset_no_water=dataset_no_water.addBands(unit_lst.rename('LST').toFloat());

var bands = ["Wet","NDVI","NDBSI","LST"];
var sentImage =dataset_no_water.select(bands);


var image =  sentImage.select(bands);
var scale = 1000;
var bandNames = image.bandNames();
// image bands rename
var getNewBandNames = function(prefix) {
    var seq = ee.List.sequence(1, bandNames.length());
    return seq.map(function(b) {
      return ee.String(prefix).cat(ee.Number(b).int());
    });
  };


//mean
var meanDict = image.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: table,
    scale: scale,
    maxPixels: 1e9
});
var means = ee.Image.constant(meanDict.values(bandNames));
var centered = image.subtract(means);


//PCA function
var getPrincipalComponents = function(centered, scale, region) {
    // Converting images to one-dimensional arrays
    var arrays = centered.toArray();

    // calculate covariance
    var covar = arrays.reduceRegion({
      reducer: ee.Reducer.centeredCovariance(),
      geometry: table,
      scale: scale,
      maxPixels: 1e9
    });
    var covarArray = ee.Array(covar.get('array'));
    var eigens = covarArray.eigen();
    var eigenValues = eigens.slice(1, 0, 1);
   
    //Calculating principal component weights
    var eigenValuesList = eigenValues.toList().flatten();
    var total = eigenValuesList.reduce(ee.Reducer.sum());
    var percentageVariance = eigenValuesList.map(function(item) {
      return (ee.Number(item).divide(total)).multiply(100).format('%.2f');
    });
    print('Eigens',eigens);    
    print('EigenValues',eigenValues);
    print("PercentageVariance", percentageVariance);


    var eigenVectors = eigens.slice(1, 1);
    var arrayImage = arrays.toArray(1);

    var principalComponents = ee.Image(eigenVectors).matrixMultiply(arrayImage);

    var sdImage = ee.Image(eigenValues.sqrt())
      .arrayProject([0]).arrayFlatten([getNewBandNames('sd')]);


    principalComponents=principalComponents
      .arrayProject([0])
      .arrayFlatten([getNewBandNames('pc')])
      .divide(sdImage);
    return principalComponents;
  };


var pcImage = getPrincipalComponents(centered, scale, table);
var rsei_un_unit = pcImage.expression(
  'constant - pc1' , 
  {
             constant: 1,
             pc1: pcImage.select('pc1')
         });
var rsei = img_normalize(rsei_un_unit);
print(rsei);
Map.addLayer(rsei, {}, 'PCA');

var folder = 'RSEI';
Export.image.toDrive({
    image: rsei,
    description: "rsei_2020",
    fileNamePrefix: 'rsei_2020',
    folder:folder,
    scale: 1000,
    crs:'EPSG:4326',
    maxPixels: 273046074,
    region: table,
    });
