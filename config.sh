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
	echo "Installing environment.yml"
        conda env create -f environment.yml
	echo "environment created"
    elif [ "$resp" = "Y" ] || [ "$resp" = "YES" ]; then
	echo "Installing environment_gpu.yml"
        conda env create -f environment_gpu.yml
	echo "environment created"
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
cd ../..
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
cd ../..
echo "Sentry-Assistant creation done!"

echo "Starting validation phase..."

cd Hybrid-v1
if ! [ -d "face_dataset" ]; then
  echo "Folder /face_dataset does NOT exist"
fi

if ! [ -d "emotion_dataset" ]; then
  echo "Folder /emotion_dataset does NOT exist"
fi

if ! [ -d "prepared_faces_dataset" ]; then
  echo "Folder /prepared_faces_dataset does NOT exist"
fi

if ! [ -d "trainedmodel" ]; then
  echo "Folder /trainedmodel does NOT exist"
fi

cd emotion_dataset
if ! [ -d "prepared" ]; then
  echo "Folder Hybrid-v1/emotion_dataset/prepared does NOT exist"
fi
if ! [ -d "raw" ]; then
  echo "Folder Hybrid-v1/emotion_dataset/raw does NOT exist"
fi
cd raw
if ! [ -d "afraid" ]; then
  echo "Folder Hybrid-v1/emotion_dataset/raw/afraid does NOT exist"
fi
if ! [ -d "angry" ]; then
  echo "Folder Hybrid-v1/emotion_dataset/raw/angry does NOT exist"
fi
if ! [ -d "disgusted" ]; then
  echo "Folder Hybrid-v1/emotion_dataset/raw/disgusted does NOT exist"
fi
if ! [ -d "happy" ]; then
  echo "Folder Hybrid-v1/emotion_dataset/raw/happy does NOT exist"
fi
if ! [ -d "neutral" ]; then
  echo "Folder Hybrid-v1/emotion_dataset/raw/neutral does NOT exist"
fi
if ! [ -d "sad" ]; then
  echo "Folder Hybrid-v1/emotion_dataset/raw/sad does NOT exist"
fi
if ! [ -d "surprised" ]; then
  echo "Folder Hybrid-v1/emotion_dataset/raw/surprised does NOT exist"
fi
cd ../../..

cd Recognizer-v2
if ! [ -d "dataset/" ]; then
  echo "Folder /dataset does NOT exist"
fi

if ! [ -d "trainedmodel/" ]; then
  echo "Folder /trainedmodel does NOT exist"
fi
cd ..
cd Sentry-Assistant
if ! [ -d "face_dataset" ]; then
  echo "Folder /face_dataset does NOT exist"
fi

if ! [ -d "emotion_dataset" ]; then
  echo "Folder /emotion_dataset does NOT exist"
fi

if ! [ -d "prepared_faces_dataset" ]; then
  echo "Folder /prepared_faces_dataset does NOT exist"
fi

if ! [ -d "trainedmodel" ]; then
  echo "Folder /trainedmodel does NOT exist"
fi
cd emotion_dataset
if ! [ -d "prepared" ]; then
  echo "Folder Sentry-Assistant/emotion_dataset/prepared does NOT exist"
fi
if ! [ -d "raw" ]; then
  echo "Folder Sentry-Assistant/emotion_dataset/raw does NOT exist"
fi
cd raw
if ! [ -d "afraid" ]; then
  echo "Folder Sentry-Assistant/emotion_dataset/raw/afraid does NOT exist"
fi
if ! [ -d "angry" ]; then
  echo "Folder Sentry-Assistant/emotion_dataset/raw/angry does NOT exist"
fi
if ! [ -d "disgusted" ]; then
  echo "Folder Sentry-Assistant/emotion_dataset/raw/disgusted does NOT exist"
fi
if ! [ -d "happy" ]; then
  echo "Folder Sentry-Assistant/emotion_dataset/raw/happy does NOT exist"
fi
if ! [ -d "neutral" ]; then
  echo "Folder Sentry-Assistant/emotion_dataset/raw/neutral does NOT exist"
fi
if ! [ -d "sad" ]; then
  echo "Folder Sentry-Assistant/emotion_dataset/raw/sad does NOT exist"
fi
if ! [ -d "surprised" ]; then
  echo "Folder Sentry-Assistant/emotion_dataset/raw/surprised does NOT exist"
fi
cd ../../..

echo "System Ready."
fi
cd ../../..

echo "Setup is done."
