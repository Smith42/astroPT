# get the data

To get the DESI survey data first download
`gz_desi_deep_learning_catalog_friendly.csv` from
`https://doi.org/10.5281/zenodo.7786416`, then run:

```
$ mkdir raws
$ chmod +x get_desi.bash
$ bash parallel_desi.bash
```

Then wait approximately three weeks(!) If anyone has a good idea to speed up
this process without hammering the DESI servers please please make a pull
request.
