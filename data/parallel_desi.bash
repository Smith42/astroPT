#!/bin/bash

DATA=$(tail -n +2 gz_desi_deep_learning_catalog_friendly.csv)

echo $DATA | xargs -n1024 | parallel --jobs 8 --delimiter ' ' ./get_desi.bash {}
