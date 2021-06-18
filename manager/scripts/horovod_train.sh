#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES={device} horovodrun -np {num_gpus} python {train_py} --config {config_file}