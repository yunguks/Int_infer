from timeit import default_timer as timer
import torch
import torch.nn as nn
from models import vgg,layers
import torch.utils.benchmark as benchmark
import pandas as pd
import os
import argparse
import numpy as np
import random

def manual_seed(seed):
    np.random.seed(seed) #1
    random.seed(seed) #2
    torch.manual_seed(seed) #3
    torch.cuda.manual_seed(seed) #4.1
    torch.cuda.manual_seed_all(seed) #4.2
    torch.backends.cudnn.benchmark = False #5 
    torch.backends.cudnn.deterministic = True #6

@torch.no_grad()
def measure_time_host(
    model: nn.Module,
    input_tensor: torch.Tensor,
    num_repeats: int = 100,
    num_warmups: int = 10,
    synchronize: bool = True,
    continuous_measure: bool = False,
) -> float:

    result_list = []
    for _ in range(num_warmups):
        _ = model.forward(input_tensor)
    torch.cuda.synchronize()

    elapsed_time_ms = 0

    if continuous_measure:
        start = timer()
        for _ in range(num_repeats):
            _ = model.forward(input_tensor)
        if synchronize:
            torch.cuda.synchronize()
        end = timer()
        elapsed_time_ms = (end - start) * 1000

    else:
        for _ in range(num_repeats):
            start = timer()
            _ = model.forward(input_tensor)
            if synchronize:
                torch.cuda.synchronize()
            end = timer()
            elapsed_time_ms += (end - start) * 1000
            result_list.append(round((end-start)*1000,5))

    return elapsed_time_ms / num_repeats, result_list

@torch.no_grad()
def measure_time_device(
    model: nn.Module,
    input_tensor: torch.Tensor,
    num_repeats: int = 100,
    num_warmups: int = 10,
    synchronize: bool = True,
    continuous_measure: bool = False,
) -> float:

    result_list = []
    for _ in range(num_warmups):
        _ = model.forward(input_tensor)
    torch.cuda.synchronize()

    elapsed_time_ms = 0

    if continuous_measure:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(num_repeats):
            _ = model.forward(input_tensor)
        end_event.record()
        if synchronize:
            # This has to be synchronized to compute the elapsed time.
            # Otherwise, there will be runtime error.
            torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)

    else:
        for _ in range(num_repeats):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            _ = model.forward(input_tensor)
            end_event.record()
            if synchronize:
                # This has to be synchronized to compute the elapsed time.
                # Otherwise, there will be runtime error.
                torch.cuda.synchronize()
            elapsed_time_ms += start_event.elapsed_time(end_event)
            result_list.append(round(start_event.elapsed_time(end_event),5))

    return elapsed_time_ms / num_repeats, result_list


@torch.no_grad()
def run_inference(model: nn.Module,
                  input_tensor: torch.Tensor) -> torch.Tensor:

    return model.forward(input_tensor)


def main(model, input_tensor) -> None:

    num_warmups = 100
    num_repeats = 1000

    model.eval()
    model.cuda()

    # Input tensor
    input_tensor = input_tensor.cuda()

    torch.cuda.synchronize()

    # print("Latency Measurement Using CPU Timer...")
    for continuous_measure in [False]:
        for synchronize in [True]:
            try:
                latency_ms, perones_total = measure_time_host(
                    model=model,
                    input_tensor=input_tensor,
                    num_repeats=num_repeats,
                    num_warmups=num_warmups,
                    synchronize=synchronize,
                    continuous_measure=continuous_measure,
                )
                print(f"|"
                      f"Synchronization: {synchronize!s:5}| "
                      f"Continuous Measurement: {continuous_measure!s:5}| "
                      f"Latency: {latency_ms:.5f} ms| ")
            except Exception as e:
                print(f"|"
                      f"Synchronization: {synchronize!s:5}| "
                      f"Continuous Measurement: {continuous_measure!s:5}| "
                      f"Latency: N/A     ms| ")
            torch.cuda.synchronize()

    # print("Latency Measurement Using CUDA Timer...")
    for continuous_measure in [False]:
        for synchronize in [True]:
            try:
                latency_ms, perones_gpu = measure_time_device(
                    model=model,
                    input_tensor=input_tensor,
                    num_repeats=num_repeats,
                    num_warmups=num_warmups,
                    synchronize=synchronize,
                    continuous_measure=continuous_measure,
                )
                print(f"|"
                      f"Synchronization: {synchronize!s:5}| "
                      f"Continuous Measurement: {continuous_measure!s:5}| "
                      f"Latency: {latency_ms:.5f} ms| ")
            except Exception as e:
                print(f"|"
                      f"Synchronization: {synchronize!s:5}| "
                      f"Continuous Measurement: {continuous_measure!s:5}| "
                      f"Latency: N/A     ms| ")
            torch.cuda.synchronize()

    # https://pytorch.org/docs/stable/_modules/torch/utils/benchmark/utils/common.html#Measurement
    # print(f"Latency: {profile_result.mean * 1000:.5f} ms")
    return perones_total, perones_gpu


def save_result(path, result, hist_time):
    result.append(hist_time)
    if os.path.exists(path):
        old_df = pd.read_csv(path,index_col=0)
        new_df = pd.DataFrame([result], columns=['type', 'target', 'index', 'in_c', 'out_c', 'h', 'batch','hist_time'])
        df = pd.concat([old_df,new_df], ignore_index=True)
    else:
        df = pd.DataFrame([result], columns=['type', 'target', 'index', 'in_c', 'out_c', 'h', 'batch','hist_time'])

    df.to_csv(path)


def get_args():
    parser = argparse.ArgumentParser(description="Memory test")
    # model or layer
    # pytorch(int,float) conv1d
    parser.add_argument("--target",choices=['all','conv','linear'], default ='all', help="select model or layer. Default all model")
    parser.add_argument("--type", choices=['torch','int','float'], default="torch", help="select model type. Default pytorch type")
    parser.add_argument("--index", type=int, default=0, help='select layer index. conv 12, linaer 2. Default 0')
    parser.add_argument("--batch", type=int, default=1, help='select batch. Default 1.')
    parser.add_argument("--path", type=str,default="result/hist_time.csv", help="csv file path")
    return parser.parse_args()


if __name__=="__main__":
    manual_seed(42)
    args = get_args()
    info = {
        "conv": {
            "inc": [4,64,64,128,128,256,256,256,512,512,512,512,512],
            "outc": [64,64,128,128,256,256,256,512,512,512,512,512,512],
            "hw":[224,224,112,112,56,56,56,28,28,28,14,14,14]
        },
        "linear":{
            "inc": [512*7*7,4096,4096],
            "outc": [4096,4096,100],
        }
    }
    
    in_c = 0
    out_c =0
    h = 0
    batch = args.batch
    if args.target !="all":
        in_c = info[args.target]['inc'][args.index]
        out_c = info[args.target]['outc'][args.index]
        if args.target =="conv":
            h = info[args.target]['hw'][args.index]
    else:
        in_c = 4
        h = 224
    
    if args.target =="conv":
        if args.type == "float":
            layer = layers.FLOATConv2d(in_c,out_c)
            x = torch.rand((batch,h,h,in_c))
        elif args.type == "int":
            layer = layers.IntConv2d(in_c,out_c)
            x = torch.randint(-128,127,(batch,h,h,in_c), dtype=torch.int8)
        else:
            layer = nn.Conv2d(in_c,out_c,3,1,1)
            x = torch.rand((batch,in_c,h,h))
    
    elif args.target == "linear":
        if args.type == "float":
            layer = layers.FLOATLinear(in_c,out_c)
            x = torch.rand((batch,in_c))
        elif args.type == "int":
            layer = layers.IntLinear(in_c,out_c)
            x = torch.randint(-128,127,(batch,in_c), dtype=torch.int8)
        else:
            layer = nn.Linear(in_c,out_c)
            x = torch.rand((batch,in_c))
    
    else:
        if args.type== "float":
            layer = vgg.float_vgg16()
            x = torch.rand((batch,h,h,in_c))
        elif args.type =="int":
            layer = vgg.int_vgg16()
            x = torch.randint(-128,127,(batch,h,h,in_c), dtype=torch.int8)
        else:
            layer = vgg.vgg16()
            x = torch.rand((batch,in_c,h,h))

    x = x.cuda()
    layer.cuda()
    result = [args.type, args.target, args.index, in_c, out_c, h, batch]
    perones_total, perones_gpu = main(layer, x)
    perones_total.sort()
    perones_gpu.sort()
    save_result(args.path, result, perones_total)
