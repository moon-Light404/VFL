
for n_domins in 2 4 8 16
do
  for a in $(seq 0.2 0.1 0.9)
  do
    python main.py --iteration 10000 --dlr 3e-4 --pseudo_train 2 --n_domins "$n_domins" --a "$a"
  done
done