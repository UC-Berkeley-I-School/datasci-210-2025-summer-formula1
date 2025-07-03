Help me create a preprocessing pipeline to get Formula 1 data into a time series format suitable for ML classification tasks.

I am using `fastf1`, a Python library, to streamline access to F1 data.

You can think of a single race as a "session":

```python
session = fastf1.get_session(2024, 'Saudi Arabian Grand Prix', 'R')
session.load()
```

Once loaded, you can access lap data, positional data, car data, and weather data.

**Lap data** is available via `session.laps` and contains the following columns of data (one row per driver and lap):
* Time (pandas.Timedelta): Session time at which the lap was set (i.e. finished)
* LapTime (pandas.Timedelta): Lap time of the last finished lap (the lap in this row)
* Driver (str): Driver number
* NumberOfLaps (int): Number of laps driven by this driver including the lap in this row
* NumberOfPitStops (int): Number of pit stops of this driver
* PitInTime (pandas.Timedelta): Session time at which the driver entered the pits. Consequently, if this value is not NaT the lap in this row is an inlap.
* PitOutTime (pandas.Timedelta): Session time at which the driver exited the pits. Consequently, if this value is not NaT, the lap in this row is an outlap.
* Sector1/2/3Time (pandas.Timedelta): Sector times (one column for each sector time)
* Sector1/2/3SessionTime (pandas.Timedelta): Session time at which the corresponding sector time was set (one column for each sector's session time)
* SpeedI1/I2/FL/ST: Speed trap speeds; FL is speed at the finish line; I1 and I2 are speed traps in sector 1 and 2 respectively; ST maybe a speed trap on the longest straight (?)

Weather data, positional data, and car data are delineated by car number and thus can be retrieved from each row of `session.laps`:

```python
session.laps.iloc[0].get_weather_data()
session.laps.iloc[0].get_car_data()
session.laps.iloc[0].get_pos_data()
```

**Car data**:
* Time (pandas.Timedelta): session timestamp (time only); inaccurate, has duplicate values; use Date instead
* Date (pandas.Timestamp): timestamp for this sample as Date + Time; more or less exact
* Speed (float): Car speed [km/h]
* RPM (float): Car RPM
* nGear (int): Car gear number
* Throttle (float): 0-100 Throttle pedal pressure [%]
* Brake (bool): Brakes are applied or not.
* DRS (int): 0-14 (Odd DRS is Disabled, Even DRS is Enabled?) (More Research Needed?)
  * 0 = Off
  * 1 = Off
  * 2 = (?)
  * 3 = (?)
  * 8 = Detected, Eligible once in Activation Zone (Noted Sometimes)
  * 10 = On (Unknown Distinction)
  * 12 = On (Unknown Distinction)
  * 14 = On (Unknown Distinction)
* Source (str): Indicates the source of a sample; 'car' for all values here

The data stream has a sample rate of (usually) 240ms. The samples from the data streams for position data and car data do not line up. Resampling/interpolation is required to merge them.

**Position data**:
* Time (pandas.Timedelta): session timestamp (time only); inaccurate, has duplicate values; use Date instead
* Date (pandas.Timestamp): timestamp for this sample as Date + Time; more or less exact
* X (float): X position [1/10 m]
* Y (float): Y position [1/10 m]
* Z (float): Z position [1/10 m]
* Status (str): Flag - OffTrack/OnTrack
* Source (str): Indicates the source of a sample; 'pos' for all values here

The data stream has a sample rate of (usually) 220ms. The samples from the data streams for position data and car data do not line up. Resampling/interpolation is required to merge them.

**For both of the above**:
* Time (timedelta): Time (0 is start of the data slice)
* SessionTime (timedelta): Time elapsed since the start of the session
* Date (datetime): The full date + time at which this sample was created
* Source (str): Flag indicating how this sample was created:
  * 'car': sample from original api car data
  * 'pos': sample from original api position data
  * 'interpolated': this sample was artificially created; all values are computed/interpolated

**Weather data** provides the following data channels per sample:
* Time (datetime.timedelta): session timestamp (time only)
* AirTemp (float): Air temperature [°C]
* Humidity (float): Relative humidity [%]
* Pressure (float): Air pressure [mbar]
* Rainfall (bool): Shows if there is rainfall
* TrackTemp (float): Track temperature [°C]
* WindDirection (int): Wind direction [°] (0°-359°)
* WindSpeed (float): Wind speed [m/s]

Weather data is updated once per minute.

Telemetry can also be retrieved directly from the `session` object, parameterized by the driver number:

```python
driverNumber = '1'
all_pos_data_for_driver_one = session.pos_data[driverNumber] # shape (46559, 8)
all_car_data_for_driver_one = session.car_data[driverNumber] # shape (45060, 10)
```

Weather data is the only telemetry type not delineated by driver:

```python
session.weather_data # shape (201, 8)
```

`fastf1` also supports a Lap method, `get_telemetry`:

```python
session.laps.iloc[0].get_telemetry(frequency)
```

Telemetry data is the result of merging the returned data from `get_car_data()` and `get_pos_data()`. This means that telemetry data at least partially contains interpolated values! Telemetry data additionally already has computed channels added (e.g. Distance). This method is provided for convenience and compatibility reasons. But using it does usually not produce the most accurate possible result. It is recommended to use `get_car_data()` or `get_pos_data()` when possible. This is also faster if merging of car and position data is not necessary and if not all computed channels are needed. Resampling during merging is done according to the frequency set by `TELEMETRY_FREQUENCY`.

**TELEMETRY_FREQUENCY = 'original'**
* Defines the frequency used when resampling the telemetry data. Either the string `'original'` or an integer to specify a frequency in Hz.

The last thing to note for `fastf1` is how timing works. A detailed explanation is pasted. The important thing to note is that each of the telemetry streams are not synchronized, and each has a distinct time interval, so we need to be careful when aggregating them.

Through merging/slicing it is possible to obtain any combination of telemetry channels! The following additional computed data channels can be added:
* Distance driven between two samples: `add_differential_distance()`
* Distance driven since the first sample: `add_distance()`
* Relative distance driven since the first sample: `add_relative_distance()`
* Distance to driver ahead and car number of said driver: `add_driver_ahead()`

```python
add_differential_distance(drop_existing=True)[source]
# Add column 'DifferentialDistance' to self.
# This column contains the distance driven between subsequent samples.
# Calls calculate_differential_distance() and joins the result with self.
# PARAMETERS:
# drop_existing (bool) – Drop and recalculate column if it already exists
# RETURN TYPE: Telemetry
# RETURNS: self joined with new column or self if column exists and drop_existing is False.

add_distance(drop_existing=True)[source]
# Add column 'Distance' to self.
# This column contains the distance driven since the first sample of self in meters.
# The data is produced by integrating the differential distance between subsequent laps. 
# You should not apply this function to telemetry of many laps simultaneously to reduce 
# integration error. Instead apply it only to single laps or few laps at a time!
# Calls integrate_distance() and joins the result with self.
# PARAMETERS: drop_existing (bool) – Drop and recalculate column if it already exists
# RETURN TYPE: Telemetry
# RETURNS: self joined with new column or self if column exists and drop_existing is False.

add_relative_distance(drop_existing=True)[source]
# Add column 'RelativeDistance' to self.
# This column contains the distance driven since the first sample as a floating point 
# number where 0.0 is the first sample of self and 1.0 is the last sample.
# This is calculated the same way as 'Distance' (see: add_distance()). 
# The same warnings apply.
# PARAMETERS: drop_existing (bool) – Drop and recalculate column if it already exists
# RETURN TYPE: Telemetry
# RETURNS: self joined with new column or self if column exists and drop_existing is False.

add_driver_ahead(drop_existing=True)[source]
# Add column 'DriverAhead' and 'DistanceToDriverAhead' to self.
# DriverAhead: Driver number of the driver ahead as string 
# DistanceToDriverAhead: Distance to next car ahead in meters
# Note: Cars in the pit lane are currently not excluded from the data. They will show up 
# when overtaken on pit straight even if they're not technically in front of the car. 
# A fix for this is TBD with other improvements.
# This should only be applied to data of single laps or few laps at a time to reduce 
# integration error. For longer time spans it should be applied per lap and the laps 
# should be merged afterwards. If you absolutely need to apply it to a whole session, 
# use the legacy implementation. Note that data of the legacy implementation will be 
# considerably less smooth. (see fastf1.legacy)
# Calls calculate_driver_ahead() and joins the result with self.
# PARAMETERS: drop_existing (bool) – Drop and recalculate column if it already exists
# RETURN TYPE: Telemetry
# RETURNS: self joined with new column or self if column exists and drop_existing is False.
```

Lastly, it's worth mentioning the `merge_channels(other, frequency=None)` method, which supports merging telemetry objects containing different telemetry channels. The two objects don't need to have a common time base. The data will be merged, optionally resampled and missing values will be interpolated.

`Telemetry.TELEMETRY_FREQUENCY` determines if and how the data is resampled. This can be overridden using the frequency keyword for this method.

Merging and resampling:
* If the frequency is 'original', data will not be resampled. The two objects will be merged and all timestamps of both objects are kept. Values will be interpolated so that all telemetry channels contain valid data for all timestamps. This is the default and recommended option.
* If the frequency is specified as an integer in Hz the data will be merged as before. After that, the merged time base will be resampled from the first value on at the specified frequency. Afterward, the data will be interpolated to fit the new time base. This means that usually most if not all values of the data will be interpolated values. This is detrimental for overall accuracy.

Interpolation:
* Missing values after merging will be interpolated for all known telemetry channels using `fill_missing()`. Different interpolation methods are used depending on what kind of data the channel contains. For example, forward fill is used to interpolated 'nGear' while linear interpolation is used for 'RPM' interpolation.

Our preprocessor needs to be architected in such a way that allows us to construct different types of datasets; for example:
* a dataset containing session data from all drivers in every race in a given season
* a dataset containing session data from all drivers in specific races
* a dataset containing session data from all drivers in one race
* a dataset containing session data from one driver in every race in a given season
* a dataset containing session data from one driver in specific races
* a dataset containing session data from one driver in one race

For example, I may wish to train a classifier on just one driver's worth of telemetry, and I may wish to evaluate how the model performance scales with respect to the dataset size: is one session sufficient, or do we need multiple? Similarly, how well does a model perform when fitted to multiple drivers worth of telemetry?

We will use native `fastf1` methods as well as the aeon-toolkit for preprocessing time series. I am primarily concerned with the following:

When merging data streams, we must ensure proper alignment of our samples. Their sampling intervals are different, so interpolation may be required. This is discussed in-depth above.

We may want to **rescale time series** to avoid discriminative patterns from differing levels of scale and variance. There are three ways to rescale time series:
* Normalise: subtract the mean and divide by the standard deviation to make all series have zero mean and unit variance.
* Re-center: Recentering involves subtracting the mean of each series
* Min-Max: Scale the data to be between 0 and 1

```python
# Normalise
from aeon.transformations.collection import Normalizer
normalizer = Normalizer()
X2 = normalizer.fit_transform(X)
np.round(np.mean(X2, axis=-1)[0:5], 6)
# Output:
# array([[ 0.],
#        [-0.],
#        [ 0.],
#        [-0.],
#        [-0.]])

# Re-center
from aeon.transformations.collection import Centerer
c = Centerer()
X3 = c.fit_transform(X)
np.round(np.mean(X3, axis=-1)[0:5], 6)

# Min-Max
from aeon.transformations.collection import MinMaxScaler
minmax = MinMaxScaler()
X4 = minmax.fit_transform(X)
np.round(np.min(X4, axis=-1)[0:5], 6)
```

We may want to **resize time series**. Suppose we have a collections of time series with different lengths, i.e. different number of time points. Currently, most of aeon's collection estimators (classification, clustering or regression) require equal-length time series. Those that can handle unequal length series are tagged with "capability:unequal".

If you want to use an estimator that cannot internally handle missing values, one option is to convert unequal length series into equal length. This can be done through padding, truncation or resizing through fitting a function and resampling.

```python
from aeon.classification.convolution_based import RocketClassifier
from aeon.datasets import load_basic_motions, load_japanese_vowels, load_plaid
from aeon.utils.validation import has_missing, is_equal_length, is_univariate
```

**Unequal or equal length collections time series:**
If a collection contains all equal length series, it will store the data in a 3D numpy of shape `(n_cases,n_channels, n_timepoints)`. If it is unequal length, it is stored in a list of 2D numpy arrays.
There are two basic strategies for unequal length problems:
1. Use an estimator that can internally handle missing values
2. Transform the data to be equal length by, for example, truncating or padding series

**Padding, truncating or resizing.**
We can pad, truncate or resize. By default, pad adds zeros to make all series the length of the longest, truncate removes all values beyond the length of the shortest and resize stretches or shrinks the series.

**Missing Values**
Missing values are indicated by `NaN` in numpy array. You can test whether any `aeon` data structure contains missing values using the utility function `has_missing(X)`.

There are a range of strategies for handling missing values. These include:
1. Use an estimator that internally handles missing values. It is fairly easy for some algorithms (such as decision trees) to internally deal with missing values, usually be using it as a distinct series value after discretisation. We do not yet have many estimators with this capability. Estimators that are able to internally handle missing values are tagged with `"capability:missing_values": True`.
2. Removing series with missing: this is often desirable if the train set size is large, the number of series with missing is small and the proportion of missing values for these series is high. We do not yet have a transformer for this, but it is easy to implement yourself.
3. Interpolating missing values from series: estimating the missing values from the other values in a time series is commonly done. This is often desirable if the train set size is small and the proportion of missing values is low. You can do this with the transformer `SimpleImputer`. This interpolates each series and each channel independently. So for example a mean interpolation of series with two channels `[[NaN,1.0,2.0,3.0],[-1.0,-2.0,-3.0,-4.0]]` would be `[[2.0,1.0,2.0,3.0],[-1.0,-2.0,-3.0,-4.0]]`.

I will provide additional details later for how to implement specific things (such as aggregating sessions for a specific season, or any of the aforementioned aeon-toolkit preprocessing steps). For now, your job is to:
* Propose a flexible software architecture
  * Should we use a functional approach or a class-based approach ideal?
  * How should we handle separation of concerns for things like preprocessing and scope of dataset size (single race, multi-race, whole season, etc.)?
* Please use native `fastf1` solutions where able, deferring to the aeon-toolkit only when it makes sense.
* If you have any questions related to the design, scope, or either of the APIs (fastf1 or aeon), please don't hesitate to ask!