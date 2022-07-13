# eurlex-4k, wiki10-31k, amazoncat-13k, amazon-670k, wiki-500k, amazon-3m
DATASET="amazon-670k"
wget https://archive.org/download/pecos-dataset/xmc-base/${DATASET}.tar.gz
tar -zxvf ./${DATASET}.tar.gz
cp -rf ./xmc-base ./dataset && rm -rf ./xmc-base