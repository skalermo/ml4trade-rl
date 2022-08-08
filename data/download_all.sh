mkdir -p ".data"

# download energy prices from google drive
energy_prices_link=https://drive.google.com/uc\?id\=1wU_zVNQoMtwc7XX1W3WZIbfBa5tuZ5oG\&export\=download
wget -N -O ".data/prices_pl.csv" "$energy_prices_link"

# Fill in missing rows
python data/check_prices_df.py

# download weather synoptic data from 2016-2022
weather_link=https://danepubliczne.imgw.pl/data/dane_pomiarowo_obserwacyjne/dane_meteorologiczne/terminowe/synop/
for year in {2016..2022}
do
    wget -rkN -np -nd -R "index.html*" -P ".data/weather/$year" "$weather_link/$year/"
done

unzip_output_dir=".data/weather_unzipped_flattened"
mkdir -p "$unzip_output_dir"

# unzip downloaded weather data
find . -name "*.zip" |
while read filename
do
  unzip -n -d "$unzip_output_dir" "$filename"
done
