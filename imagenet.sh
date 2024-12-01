# python main.py --dataset "tinyImagenet" --dataset_portion 0.1 --pseudo_train 2 --a 0.8 --n_domins 4 --iteration 20000 --dlr 3e-4 
# python main.py --dataset "tinyImagenet" --dataset_portion 0.1 --pseudo_train 2 --a 0.3 --n_domins 4 --iteration 20000 --dlr 3e-4

python main.py --dataset "tinyImagenet" --dataset_portion 0.1 --pseudo_train 2 --a 0.3 --n_domins 8 --iteration 15000 --dlr 3e-4
python main.py --dataset "tinyImagenet" --dataset_portion 0.1 --pseudo_train 2 --a 0.8 --n_domins 8 --iteration 15000 --dlr 3e-4


python main.py --dataset "tinyImagenet" --dataset_portion 0.1 -- batch_size 128 --pseudo_train 2 --a 0.3 --n_domins 8 --iteration 15000 --dlr 3e-4
python main.py --dataset "tinyImagenet" --dataset_portion 0.1 -- batch_size 128 --pseudo_train 2 --a 0.8 --n_domins 8 --iteration 15000 --dlr 3e-4

python main.py --dataset "tinyImagenet" --dataset_portion 0.1 -- batch_size 128 --pseudo_train 2 --a 0.3 --n_domins 16 --iteration 15000 --dlr 3e-4
python main.py --dataset "tinyImagenet" --dataset_portion 0.1 -- batch_size 128 --pseudo_train 2 --a 0.8 --n_domins 16 --iteration 15000 --dlr 3e-4