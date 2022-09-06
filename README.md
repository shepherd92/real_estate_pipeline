# Introduction

This repository attempts to predict real estate prices by executing the following steps:
1. First it scrapes real estate data from www.ingatlan.com
2. Using the provided address, it uses the nominatim API to get GPS coordinates and its uncertainty.
3. Next, it uses the Overpass API and openstreetmap to calculate localization features such as the ratio of green areas in a 100 meter radius circle.
4. Finally, using the provided features and machine learning models, it attempts to predict the prices of the real estates.

# Prerequisites

## Nominatim API

To install a local nominatim server, follow the below instructions.

1. a) Download "mediagis/nominatim-docker" docker image
   docker pull mediagis/nominatim-docker
   b) Load the docker image from tarball
   docker load -i ~/docker/nominatim_docker_40.tar.gz

2. Download the osm.pbf database from https://download.geofabrik.de/europe/hungary-latest.osm.pbf

3. Run the below command with internet connection:

docker run -it \
    --shm-size=1g \
    -e PBF_PATH=/nominatim/data/hungary-latest.osm.pbf \
    -e REPLICATION_URL=https://download.geofabrik.de/europe/hungary-updates/ \
    -e IMPORT_WIKIPEDIA=true \
    -e IMPORT_STYLE=extratags \
    -v /home/jup2bp/docker/nominatim_data:/var/lib/postgresql/12/main \
    -v /home/jup2bp/docker/osm-maps:/nominatim/data \
    -p 8080:8080 \
    --name nominatim \
    mediagis/nominatim:4.0

4. Start the container (even if the previous step fails)
   docker start nominatim

5. Attach to the container shell
   docker exec -it nominatim bash
   sudo -u postgres -i  # use postgres user
   cd /nominatim/
   nominatim special-phrases --import-from-wiki  # import special keywords from Wikipedia
   exit
   exit

## Overpass API

To install a local overpass API server, follow the below instructions.
description: https://hub.docker.com/r/wiktorn/overpass-api

1. Download "wiktorn/overpass-api" docker image
   docker pull wiktorn/overpass-api

2. Download the osm.pbf database from https://download.geofabrik.de/europe/hungary-latest.osm.pbf

3. Run the below command with internet connection:

docker run \
  -e OVERPASS_MODE=init \
  -e OVERPASS_META=yes \
  -e OVERPASS_PLANET_URL=https://download.geofabrik.de/europe/hungary-latest.osm.pbf \
  -e OVERPASS_PLANET_PREPROCESS='mv /db/planet.osm.bz2 /db/planet.osm.pbf && osmium cat -o /db/planet.osm.bz2 /db/planet.osm.pbf && rm /db/planet.osm.pbf' \
  -e OVERPASS_RULES_LOAD=50 \
  -v /home/jup2bp/docker/overpass/:/db \
  -p 12345:80 \
  -i -t \
  --name overpass \
  wiktorn/overpass-api

4. Stop and start the container
   docker stop overpass
   docker start overpass

5. Attach to the container shell
   docker exec -it overpass bash
   sudo -u postgres -i  # use postgres user
   cd /nominatim/
   nominatim special-phrases --import-from-wiki  # import special keywords from Wikipedia
   exit
   exit

To get place rank available, change the below parts in nominatim.py (2 occurrences)

   'format': 'json',
to
   'format': 'jsonv2'

# Further Notes

## Useful websites

- https://wiki.openstreetmap.org/wiki/Nominatim/Special_Phrases/EN
- https://nominatim.openstreetmap.org/
- https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL
- https://overpass-turbo.eu/

## Features to Implement

- time seen
- size of settlement (population, area)
- average price of 10 closest apartment (spatially and kNN)

## Models to Try

- kNN
- linear regression
- neural network
- random forest
- hedonic model: ln(P) = a + b1 * R + b2 * L0
  - R is the physical characteristics matrix
  - L is the location characteristics matrix

- model: https://link.springer.com/article/10.1007/s11146-010-9262-3
- features: https://www.mdpi.com/2220-9964/7/5/168/pdf
