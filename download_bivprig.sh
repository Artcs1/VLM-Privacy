#!/bin/bash
wget https://vizwiz.cs.colorado.edu/biv-priv/images/support_images.zip
wget https://vizwiz.cs.colorado.edu/biv-priv/images/query_images.zip
unzip support_images.zip -x "__MACOSX/*"
unzip query_images.zip -x "__MACOSX/*"
mkdir -p BIV-Priv_Image
mkdir -p BIV-Priv_Image/all_images
cd BIV-Priv_Image/all_images
mv ../../support_images/* ./
mv ../../query_images/* ./
rm -rf ../../support_images.zip
rm -rf ../../support_images
rm -rf ../../query_images.zip
rm -rf ../../query_images
