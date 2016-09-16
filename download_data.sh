#!/usr/bin/env bash

# This scripts downloads the prerequisites and trained TF models.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading..."

cd data
mkdir -p prereq
cd prereq
wget https://www.dropbox.com/s/mdae915tytzvmq9/googlenet.pb
wget https://www.dropbox.com/s/ludyh41agl0xuox/vgg16_weights.npz
cd ..
mkdir -p models
cd models
wget https://www.dropbox.com/s/xputod3d2591sog/gnet-fc.ckpt-6744
wget https://www.dropbox.com/s/qp6aipxyi24o0ds/vgg16.ckpt-6601

echo "Done."
