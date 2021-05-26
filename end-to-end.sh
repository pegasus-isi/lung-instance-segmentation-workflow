python bin/preprocess.py -t train -c t -i "data/LungSegmentation/CXR_png" -o output -m "data/LungSegmentation/masks"
wait $!
python bin/preprocess.py -t val -c t -i "data/LungSegmentation/CXR_png" -o output -m "data/LungSegmentation/masks"
wait $!
python bin/preprocess.py -t test -c t -i "data/LungSegmentation/CXR_png" -o output -m "data/LungSegmentation/masks"
wait $!
cp "data/LungSegmentation/masks/"* output
wait $!
python bin/hpo.py -i output -o output
wait $!
python bin/train_model.py -i output -o output
wait $!
python bin/prediction.py -i output -o output
wait $!
python bin/evaluate.py -i output -o output
wait $!