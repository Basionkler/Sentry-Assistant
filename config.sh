#!/bin/bash

clear
echo "Starting env setup..."

valid_input=false

until $valid_input; do
    read -r -p "have you an Nvidia gpu with CUDA support? (y/n): " resp

    # Trim the input and translate it to uppercase.
    resp="$(echo "$resp" | tr /a-z/ /A-Z/)"

    valid_input=true

    if [ "$resp" = "N" ] || [ "$resp" = "NO" ]; then
        conda env create -f environment.yml
    elif [ "$resp" = "Y" ] || [ "$resp" = "YES" ]; then
        conda env create -f environment_gpu.yml
    else
        echo "Error: Select y/n"

        valid_input=false
    fi
done

echo "Starting creation for Hybrid Project..."
cd Hybrid-v1
mkdir -p face_dataset emotion_dataset prepared_faces_dataset trainedmodel
cd emotion_dataset
mkdir -p raw prepared raw/afraid raw/angry raw/disgusted raw/happy raw/neutral raw/sad raw/surprised
cd ..
echo "Hybrid Project creation done!"

echo "Starting creation for Recognizer Project..."
cd Recognizer-v2
mkdir -p dataset trainedmodel
cd ..
echo "Recognizer Project creation done!"

echo "Starting creation for Sentry-Assistant Project..."
cd Sentry-Assistant
mkdir -p face_dataset emotion_dataset prepared_faces_dataset trainedmodel
cd emotion_dataset
mkdir -p raw prepared raw/afraid raw/angry raw/disgusted raw/happy raw/neutral raw/sad raw/surprised
cd ..
echo "Sentry-Assistant creation done!"

echo "Starting validation phase..."

DIRECTORY = "/Hybrid-v1"
if [! -d "${DIRECTORY}/face_dataset"]; then
  echo "Folder ${DIRECTORY}/face_dataset does NOT exist"
fi

if [! -d "${DIRECTORY}/emotion_dataset" ]; then
  echo "Folder ${DIRECTORY}/emotion_dataset does NOT exist"
fi

if [! -d "${DIRECTORY}/prepared_faces_dataset" ]; then
  echo "Folder ${DIRECTORY}/prepared_faces_dataset does NOT exist"
fi

if [! -d "${DIRECTORY}/trainedmodel" ]; then
  echo "Folder ${DIRECTORY}/trainedmodel does NOT exist"
fi

DIRECTORY = "/Recognizer-v2"
if [! -d "${DIRECTORY}/dataset"]; then
  echo "Folder ${DIRECTORY}/dataset does NOT exist"
fi

if [! -d "${DIRECTORY}/trainedmodel" ]; then
  echo "Folder ${DIRECTORY}/trainedmodel does NOT exist"
fi

DIRECTORY = "/Sentry-Assistant"
if [! -d "${DIRECTORY}/face_dataset"]; then
  echo "Folder ${DIRECTORY}/face_dataset does NOT exist"
fi

if [! -d "${DIRECTORY}/emotion_dataset" ]; then
  echo "Folder ${DIRECTORY}/emotion_dataset does NOT exist"
fi

if [! -d "${DIRECTORY}/prepared_faces_dataset" ]; then
  echo "Folder ${DIRECTORY}/prepared_faces_dataset does NOT exist"
fi

if [! -d "${DIRECTORY}/trainedmodel" ]; then
  echo "Folder ${DIRECTORY}/trainedmodel does NOT exist"
fi

echo "Setup is done."
