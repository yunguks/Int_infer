#!/bin/bash

run_time=false
run_memory=false
run_hist=false

# 명령행 인수에 따라 변수 설정
if [ "$1" == "time" ]; then
    run_time=true
elif [ "$1" == "memory" ]; then
    run_memory=true
elif [ "$1" == "all" ]; then
    run_time=true
    run_memory=true
elif [ "$1" == "hist" ]; then
    run_hist=true
else
    echo "time and memory experiment x. Usage: $0 {time|memory|all}"
fi

types=("torch" "int" "float")
batch=(64)
# 선택된 실험만 실행
for t in "${types[@]}"; do
    for b in "${batch[@]}"; do
        if [ "$run_time" = true ]; then
            python check_time.py --target all --type $t --batch $b
            sleep 1m
            for var in {0..12}; do
                python check_time.py --target conv --type $t --index $var --batch $b
                sleep 1m
            done

            for var in {0..2}; do
                python check_time.py --target linear --type $t --index $var --batch $b
                sleep 1m
            done
        fi

        if [ "$run_memory" = true ]; then
            python check_memory.py --target all --type $t --batch $b
            sleep 1m

            for var in {0..12}; do
                python check_memory.py --target conv --type $t --index $var --batch $b
                sleep 1m
            done

            for var in {0..2}; do
                python check_memory.py --target linear --type $t --index $var --batch $b
                sleep 1m
            done
        fi
        

    done
done

for t in "${types[@]}"; do
    if [ "$run_hist" = true ]; then
        python hist_time.py --target all --type $t --batch $b

        for var in {0..12}; do
            python hist_time.py --target conv --type $t --index $var --batch $b
        done

        for var in {0..2}; do
            python hist_time.py --target linear --type $t --index $var --batch $b
        done
    fi
done

lr=(1e-4 7e-5 5e-5 2e-5)
if [ "$1" == "train" ]; then
    for l in "${lr[@]}"; do
        python train.py --lr $l --optim sgd
        python train.py --lr $l --optim adam
    done
fi

