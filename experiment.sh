#!/bin/bash

# time exp
types=("torch" "int" "float")
for t in "${types[@]}";
do
    python check_time.py --target all --type $t

    for var in {0..12};
    do
        python check_time.py --target conv --type $t --index $var
    done

    for var in {0..2};
    do
        python check_time.py --target linear --type $t --index $var
    done
done