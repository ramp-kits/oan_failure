# Optical Access Network Challenge

## Preparing data

The raw data is located in the subfolder `/data`. It consists in a set of a few thousand time series that spear around 20 days of time.

```
data/
    city_A/
        source.npy
        source_labels.npy
    city_B/
        target.npy
        target_labels.npy
        test.npy
        test_labels.npy
```

However, the challenge consists in predicting failure using only a single week of data.
Hence, we preprocess the original data with the following code
```
~/hackathon/data $ python prepare_data.py
```
This program will consider the original data and will output a `pickle` file that will generate sub time-series of a 1 week length.
The output will be located as follow
```data/
    city_A/
        ramp_train.pickle
    city_B/
        ramp_target.pickle
        ramp_test.pickle
```
These file are the one that will be read by `problem.py`.

Be careful: we leave you the original data so you can explore it. But the private dataset has been generated using the original `prepare_data.py` program.
So mess with it, we do not forget how the private dataset is processed :)
