import os
import re
import sys
import glob
import json
import shutil
import mlflow
import logging
import subprocess

import hydra
from hydra import utils
from omegaconf import DictConfig, OmegaConf

from mlflow import log_metric, log_param, log_artifacts, log_artifact
from utils import log_params_from_omegaconf_dict, infer_gpu

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts')

#################### Capture Metrics from Std Output ####################

p1 = r'.*Step (\d+): start running validation on (val|test) split...'
p2 = r'.*score: (0\.\d+|[1-9]\d*\.\d+)'
p3 = r'.*Step (\d+): loss=(0\.\d+|[1-9]\d*\.\d+)'

step = 0
split = 'val'
def maybe_log_metrics(textline):
    if isinstance(textline, bytes):
        textline = textline.decode('utf8')
    global step
    global split
    res1 = re.match(p1, textline)
    res2 = re.match(p2, textline)
    res3 = re.match(p3, textline)
    if res1:
        step = int(res1[1])
        split = res1[2]
    if res2:
        mlflow.log_metric(f'score-{split}', float(res2[1]), step=step)
    if res3:
        mlflow.log_metric(f'train_loss', float(res3[2]), step=int(res3[1]))

#################### Main ####################

logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="config")
def main(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    
    mlflow_cfg = cfg.mlflow
    exp_cfg = cfg.configs
    scripts_cfg = cfg.scripts
    
    mlflow.set_tracking_uri('file://' + mlflow_cfg.output_dir + '/mlruns')
    mlflow.set_experiment(mlflow_cfg.exp_name)
    
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        save_dir = run.info.artifact_uri.replace('file://', '')
        
        # real output_dir
        exp_cfg.output_dir = exp_cfg.output_dir or os.path.join(mlflow_cfg.output_dir, 'exp_results')
        exp_cfg.output_dir = os.path.join(exp_cfg.output_dir, mlflow_cfg.task+'-'+run_id)
        
        # save configs
        log_params_from_omegaconf_dict(exp_cfg)
        config_file = os.path.join(save_dir, 'config.json')
        json.dump(OmegaConf.to_container(exp_cfg, resolve=True), open(config_file,'w'))
        
        device = infer_gpu(require_n=mlflow_cfg.num_gpus, no_wait=True)
        train_cmd = open(scripts_cfg.train_sh).read()
        train_cmd = train_cmd.format(device=device, train_py=scripts_cfg.train_py, config_file=config_file)
        
        # save all code files
        for path in mlflow_cfg.code_dirs:
            if os.path.isdir(path):
                log_artifacts(path, artifact_path=f"code/{os.path.basename(path)}")
            elif os.path.isfile(path):
                log_artifact(path, artifact_path="code")
        
        # save cmds
        cmd_file = os.path.join(save_dir, 'train_cmd.sh')
        with open(cmd_file, 'w') as fout:
            fout.write(train_cmd)
        
        # run cmd
        if mlflow_cfg.debug:
            p = subprocess.run(train_cmd, shell=True)
            print(f'rm -rf {exp_cfg.output_dir}')
            print(f'rm -rf {os.path.dirname(save_dir)}')
            # shutil.rmtree(exp_cfg.output_dir)
            # shutil.rmtree(os.path.dirname(save_dir))
        else:
            log_file = os.path.join(save_dir, 'log.txt')
            log_output = open(log_file, 'wb')
            p = subprocess.Popen(train_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            while p.poll() is None:
                line = p.stdout.readline()
                maybe_log_metrics(line) # capture metrics
                if mlflow_cfg.verbose:
                    print(line)
                log_output.write(line)
            if p.returncode == 0:
                log_output.close()
                # for json_file in glob.glob(os.path.join(cfg.output_dir, '*.json')):
                #     log_artifact(json_file, artifact_path="results")
                logger.info('Training success')


if __name__ == "__main__":
    main()