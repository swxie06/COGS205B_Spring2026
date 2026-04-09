#!/usr/bin/env bash
wget -O data.zip 'https://github.com/joachimvandekerckhove/cogs205b-s26/raw/refs/heads/main/modules/02-version-control/files/data.zip' 
mkdir -p /tmp/extracted_data
unzip -q data.zip -d /tmp/extracted_data
DATE=$(date +%F)
mkdir -p ./data/$DATE
mv /tmp/extracted_data/*.csv /data/$DATE/
git add data/ 
git commit -m "Add csv data"
git add scripts/fetch-csvs.sh
git commit -m "Add fetch-csvs script"
git remote add origin https://github.com/swxie06/COGS205B_Spring2026.git
git push -u origin main