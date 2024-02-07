#!/bin/bash

ALREADY_FETCHED=$(ls raws/*.jpg | wc -l)
DATA=$(tail -n +2 gz_desi_deep_learning_catalog_friendly.csv)

echo $DATA | xargs -n1024 | parallel --delay 0.25 --jobs 16 --delimiter ' ' ./get_desi.bash {}
