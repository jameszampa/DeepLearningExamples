mkdir -p output
python3.8 -m multiproc train.py -m Tacotron2 -o ./output/ -lr 1e-3 --epochs 2000 -bs 24 --weight-decay 1e-6 --grad-clip-thresh 1.0 --cudnn-enabled --log-file nvlog.json --anneal-steps 500 1000 1500 --anneal-factor 0.1 --resume-from-last
