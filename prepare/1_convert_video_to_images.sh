#!/bin/bash

# convert the mp4 video to images
#   Usage (sudo for the remove priviledge):
#       sudo ./convert_video_to_images.sh path/to/video fps
#   Example Usage:
#       sudo ./convert_video_to_images.sh ~/document/videofile/ 5
#   Example Output:
#       ~/document/videofile/walk/video1.mp4 
#       #=>
#       ~/document/videofile/walk/video1/00001.jpg
#       ~/document/videofile/walk/video1/00002.jpg
#       ~/document/videofile/walk/video1/00003.jpg
#       ~/document/videofile/walk/video1/00004.jpg
#       ~/document/videofile/walk/video1/00005.jpg
#       ...

for folder in $1/*
do
    for file in "$folder"/*.mp4
    do
        if [[ ! -d "${file[@]%.mp4}" ]]; then
            mkdir -p "${file[@]%.mp4}"
        fi
        ffmpeg -i "$file" -vf fps=$2 -qscale:v 1 "${file[@]%.mp4}"/%05d.jpg
        rm "$file"
    done
done