#!/bin/bash

output_file="annotations.json"
label_folder="labels"

echo '{ "type": "instances", "annotations": [' > $output_file

annotation_id=1

for label_file in $(ls $label_folder/*.txt | sort); do
  base_name=$(basename "$label_file" .txt)
  image_id=$(echo "$base_name" | sed 's/^0*//')

  while IFS= read -r line || [[ -n "$line" ]]; do
    label=($line)
    category_id=${label[0]}
    x_left=${label[1]}
    y_top=${label[2]}
    width=${label[3]}
    height=${label[4]}
    iscrowd=0

    img_width=640
    img_height=640
    x=$(echo "$x_left $img_width" | awk '{print $1 * $2}')
    y=$(echo "$y_top $img_height" | awk '{print $1 * $2}')
    bbox_width=$(echo "$width $img_width" | awk '{print $1 * $2}')
    bbox_height=$(echo "$height $img_height" | awk '{print $1 * $2}')

    echo '  {' >> $output_file
    echo '    "image_id": '"$image_id," >> $output_file
    echo '    "bbox": [' >> $output_file
    echo '      '"$x," >> $output_file
    echo '      '"$y," >> $output_file
    echo '      '"$bbox_width," >> $output_file
    echo '      '"$bbox_height" >> $output_file
    echo '    ],' >> $output_file
    echo '    "category_id": '"$category_id," >> $output_file
    echo '    "id": '"$annotation_id," >> $output_file
    echo '    "iscrowd": '"$iscrowd" >> $output_file
    echo '  },' >> $output_file

    annotation_id=$((annotation_id + 1))
  done < "$label_file"
done

sed -i '' '$ s/,$//' $output_file
echo '] }' >> $output_file

echo "JSON File Generated: $output_file"
