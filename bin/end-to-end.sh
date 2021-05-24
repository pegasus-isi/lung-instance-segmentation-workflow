python bin/preprocess.py -t train -c t -i img/lung-images/ -o output/ -m img/lung-masks/
wait $!
python bin/preprocess.py -t val -c t -i img/lung-images/ -o output/ -m img/lung-masks/
wait $!
python bin/preprocess.py -t test -c t -i img/lung-images/ -o output/ -m img/lung-masks/
wait $!
python bin/hpo.py -i output -o output
wait $!
python bin/train_model.py -i output -o output
wait $!
python bin/prediction.py -i output -o output
wait $!
python bin/evaluate.py -i output -o output
wait $!
