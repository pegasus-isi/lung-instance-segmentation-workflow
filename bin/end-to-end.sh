python bin/preprocess.py -t train -c t -i img/lung-images/ -o test-op/ -m img/lung-masks/
wait $!
python bin/preprocess.py -t val -c t -i img/lung-images/ -o test-op/ -m img/lung-masks/
wait $!
python bin/preprocess.py -t test -c t -i img/lung-images/ -o test-op/ -m img/lung-masks/
wait $!
python bin/hpo.py -i test-op -o test-op
wait $!
python bin/train_model.py -i test-op -o test-op
wait $!
python bin/prediction.py -i test-op -o test-op
wait $!
python bin/evaluate.py -i test-op -o test-op
wait $!
