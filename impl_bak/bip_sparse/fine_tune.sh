python main.py --decay=1e-4 --lr=0.001 --layer=3 --keepprob=0.8 --seed=2020 --dataset="gowalla" --topks="[20]" --bpr_batch=10000 --dropout=0 --epochs=1500
python main.py --decay=5e-4 --lr=0.001 --layer=3 --keepprob=0.8 --seed=2020 --dataset="gowalla" --topks="[20]" --bpr_batch=10000 --dropout=0 --epochs=1000
python main.py --decay=1e-4 --lr=0.002 --layer=3 --keepprob=0.8 --seed=2020 --dataset="gowalla" --topks="[20]" --bpr_batch=10000 --dropout=0 --epochs=1000
python main.py --decay=1e-4 --lr=0.001 --layer=4 --keepprob=0.8 --seed=2020 --dataset="gowalla" --topks="[20]" --bpr_batch=10000 --dropout=0 --epochs=1000
python main.py --decay=1e-4 --lr=0.001 --layer=3 --keepprob=0.6 --seed=2020 --dataset="gowalla" --topks="[20]" --bpr_batch=10000 --dropout=1 --epochs=1000
python main.py --decay=1e-4 --lr=0.001 --layer=3 --keepprob=0.8 --seed=0 --dataset="gowalla" --topks="[20]" --bpr_batch=10000 --dropout=0 --epochs=1000
python main.py --decay=1e-4 --lr=0.001 --layer=3 --keepprob=0.8 --seed=2020 --dataset="gowalla" --topks="[20]" --bpr_batch=10000 --dropout=1 --epochs=1000 --twohop=8e-3
python main.py --decay=1e-4 --lr=0.001 --layer=3 --keepprob=0.8 --seed=2020 --dataset="gowalla" --topks="[20]" --bpr_batch=10000 --dropout=0 --epochs=1000 --twohop=2e-3
