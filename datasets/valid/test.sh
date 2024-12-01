#!/bin/bash

output_file="output.json"
image_folder="images"

echo '{ "images": [' > $output_file

for img in $(ls $image_folder/*.jpg | sort); do
  file_name=$(basename "$img")

  id=$(echo "$file_name" | grep -o '[0-9]*')
  image_id=$(echo "$id" | sed 's/^0*//')

  width=640
  height=640

  echo '  {' >> $output_file
  echo '    "file_name": "'"$file_name"'",' >> $output_file
  echo '    "id": '"$image_id," >> $output_file
  echo '    "height": '"$height," >> $output_file
  echo '    "width": '"$width" >> $output_file
  echo '  },' >> $output_file
done

sed -i '$ s/,$//' $output_file
echo '] }' >> $output_file

echo "JSON file Generated: $output_file"
