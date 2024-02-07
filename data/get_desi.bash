#!/bin/bash

echo $1

NAME=$(echo $1 | cut -d',' -f1)
RA=$(echo $1 | cut -d',' -f2)
DEC=$(echo $1 | cut -d',' -f3)
echo Dumping $NAME $RA $DEC $Z
wget -nc -O raws/${NAME}.jpg "http://legacysurvey.org/viewer/jpeg-cutout?ra=$RA&dec=$DEC&size=512&layer=ls-dr8&pixscale=0.262"
#wget -nc -O raws/${NAME}.fits "http://legacysurvey.org/viewer/fits-cutout?ra=$RA&dec=$DEC&size=512&layer=ls-dr8&pixscale=0.262&bands=grz"
echo $NAME >> done.txt
