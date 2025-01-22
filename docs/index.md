# Time Series Forecasting Evaluation and Deployment (TimeFED)

TimeFED is a machine learning system for Time series Forecasting, Evaluation and Deployment.

Prediction and forecasting require building models from historical data in order to predict future observations.  The most well-known forecast types are related to weather, finance (e.g., stock market) and business (e.g., earnings prediction); however, nearly all industries and research disciplines need to perform some type of prediction on time series data, which is data that is regularly observed in time.  

Many existing software packages require data to be machine learning (ML) ready.  For time series data, that means there are no missing observations in the data streams, or missing observations can be interpolated easily using standard techniques.  Also, packages tend to assume one of two types of data:
* A continuous stream of data (referred to as data streams).  Usually there are several streams of data collecting in time together.  We refer to this type of data are multivariate time series streams.
* A set of time series (may be multivariate) with distinct start and end times.  We refer to this data as multivariate time series tracks.

TimeFED was created in response to the following data realities:

* Data contains significant gaps (sometimes on the order of months or years) due to sensor outages
* Data are not sampled at the same rates
* Time series data can be in stream or track form

The Machine Learning and Instrument Autonomy group at JPL has built an infrastructure for time series prediction and forecasting that respects these realities.
