# train
python my_train_MTNet.py --lambda_tri_sup 0.06 --exp DAMATN-FDF --consistency 0.65 --labelnum 8 --max_iterations 6000 && \
python my_train_MTNet.py --lambda_tri_sup 0.06 --exp DAMATN-FDF --consistency 0.4 --labelnum 16 --max_iterations 6000

# test
# python my_test_LA.py --model DAMATN-FDF