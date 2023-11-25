#!/bin/bash

run_time=false
run_memory=false

# 명령행 인수에 따라 변수 설정
if [ "$1" == "time" ]; then
    run_time=true
elif [ "$1" == "memory" ]; then
    run_memory=true
elif [ "$1" == "all" ]; then
    run_time=true
    run_memory=true
else
    echo "Invalid argument. Usage: $0 {time|memory|all}"
    exit 1
fi

types=("torch" "int" "float")

# 선택된 실험만 실행
for t in "${types[@]}"; do
    if [ "$run_time" = true ]; then
        python check_time.py --target all --type $t

        for var in {0..12}; do
            python check_time.py --target conv --type $t --index $var
        done

        for var in {0..2}; do
            python check_time.py --target linear --type $t --index $var
        done
    fi

    if [ "$run_memory" = true ]; then
        python check_memory.py --target all --type $t

        for var in {0..12}; do
            python check_memory.py --target conv --type $t --index $var
        done

        for var in {0..2}; do
            python check_memory.py --target linear --type $t --index $var
        done
    fi
done