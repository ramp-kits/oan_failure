# Optical Access Network Challenge

## Preparing data

After unzipping the file given to you in the Slack team, raw data is located in the subfolder `/data`. It consists in a set of a few thousand time series that span about 20 days of time.

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
This program will take the original data and will output a `pickle` file that will generate one-week-long sub time-series.
The output will be as follows:
```data/
    city_A/
        ramp_train.pickle
    city_B/
        ramp_target.pickle
        ramp_test.pickle
```
These are the files that will be read by `problem.py`.

Be careful: we give you the original data so you can explore it so you can do other transformations if you'd like. But note that the private data has been generated using the original `prepare_data.py` program.
