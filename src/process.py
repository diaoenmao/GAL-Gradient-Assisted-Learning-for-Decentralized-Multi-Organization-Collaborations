import os
import itertools
import json
import numpy as np
import shutil
from config import cfg
from utils import save, load, makedir_exist_ok

model_path = './output/model'
result_path = './output/result'
backup_path = './output/backup'
num_experiments = 12
exp = [str(x) for x in list(range(num_experiments))]
data_names = ['CIFAR10', 'CIFAR100', 'COIL100', 'Omniglot']
base_metrics = {'cvae': 'test/BCE', 'mcvae': 'test/BCE', 'vqvae': 'test/MSE', 'cpixelcnn': 'test/NLL',
                'mcpixelcnn': 'test/NLL', 'cglow': 'test/Loss', 'mcglow': 'test/Loss'}


def main():
    cvae_control = [exp, data_names, ['label'], ['cvae']]
    mcvae_control = [exp, data_names, ['label'], ['mcvae'], ['0.5']]
    cpixelcnn_control = [exp, data_names, ['label'], ['cpixelcnn']]
    mcpixelcnn_control = [exp, data_names, ['label'], ['mcpixelcnn'], ['0.5']]
    cglow_control = [exp, data_names, ['label'], ['cglow']]
    mcglow_control = [exp, data_names, ['label'], ['mcglow'], ['0.5']]
    cgan_control = [exp, data_names, ['label'], ['cgan']]
    mcgan_control = [exp, data_names, ['label'], ['mcgan'], ['0.5']]
    controls_list = [cvae_control, mcvae_control, cpixelcnn_control, mcpixelcnn_control, cglow_control, mcglow_control,
                     cgan_control, mcgan_control]
    controls = []
    for i in range(len(controls_list)):
        controls.extend(list(itertools.product(*controls_list[i])))
    processed_result = process_result(controls)
    with open('{}/processed_result.json'.format(result_path), 'w') as fp:
        json.dump(processed_result, fp, indent=2)
    save(processed_result, os.path.join(result_path, 'processed_result.pt'))
    make_vis(processed_result)
    return


def process_result(controls):
    processed_result = {}
    for control in controls:
        model_tag = '_'.join(control)
        extract_result(list(control), processed_result, model_tag)
    summarize_result(processed_result)
    return processed_result


def extract_result(control, processed_result, model_tag):
    if len(control) == 1:
        exp_idx = exp.index(control[0])
        base_result_path_i = os.path.join(result_path, '{}.pt'.format(model_tag))
        is_result_path_i = os.path.join(result_path, 'is_generated_{}.npy'.format(model_tag))
        fid_result_path_i = os.path.join(result_path, 'fid_generated_{}.npy'.format(model_tag))
        dbi_result_path_i = os.path.join(result_path, 'dbi_created_{}.npy'.format(model_tag))
        if os.path.exists(base_result_path_i):
            if 'base' not in processed_result:
                processed_result['base'] = {'exp': [None for _ in range(num_experiments)]}
            base_result = load(base_result_path_i)
            model_name = model_tag.split('_')[3]
            processed_result['base']['exp'][exp_idx] = base_result['logger'].mean[base_metrics[model_name]]
        if os.path.exists(is_result_path_i):
            if 'is' not in processed_result:
                processed_result['is'] = {'exp': [None for _ in range(num_experiments)]}
            is_result = load(is_result_path_i, mode='numpy')
            processed_result['is']['exp'][exp_idx] = is_result.item()
        if os.path.exists(fid_result_path_i):
            if 'fid' not in processed_result:
                processed_result['fid'] = {'exp': [None for _ in range(num_experiments)]}
            fid_result = load(fid_result_path_i, mode='numpy')
            processed_result['fid']['exp'][exp_idx] = fid_result.item()
        if os.path.exists(dbi_result_path_i):
            if 'dbi' not in processed_result:
                processed_result['dbi'] = {'exp': [None for _ in range(num_experiments)]}
            dbi_result = load(dbi_result_path_i, mode='numpy')
            processed_result['dbi']['exp'][exp_idx] = dbi_result.item()
    else:
        if control[1] not in processed_result:
            processed_result[control[1]] = {}
        extract_result([control[0]] + control[2:], processed_result[control[1]], model_tag)
    return


def summarize_result(processed_result):
    if 'exp' in processed_result:
        processed_result['exp'] = np.stack(processed_result['exp'], axis=0)
        processed_result['mean'] = np.mean(processed_result['exp'], axis=0).item()
        processed_result['std'] = np.std(processed_result['exp'], axis=0).item()
        processed_result['max'] = np.max(processed_result['exp'], axis=0).item()
        processed_result['min'] = np.min(processed_result['exp'], axis=0).item()
        processed_result['argmax'] = np.argmax(processed_result['exp'], axis=0).item()
        processed_result['argmin'] = np.argmin(processed_result['exp'], axis=0).item()
        processed_result['exp'] = processed_result['exp'].tolist()
    else:
        for k, v in processed_result.items():
            summarize_result(v)
    return


def make_vis(processed_result):
    s = ['#!/bin/bash\n']
    vis(s, [], processed_result)
    s = ''.join(s)
    run_file = open('./make_vis.sh', 'w')
    run_file.write(s)
    run_file.close()
    return


def vis(s, control, processed_result):
    makedir_exist_ok(model_path)
    if 'exp' in processed_result:
        data_name = control[0]
        model_name = control[2]
        metric = control[-1]
        save_per_mode = 10
        filenames = ['generate', 'transit', 'create']
        pivot = 'is'
        if metric != pivot:
            return
        best_seed = exp[processed_result['argmax']]
        for filename in filenames:
            if model_name == 'vqvae' or (filename == 'transit' and 'pixelcnn' in model_name):
                continue
            label = model_name[2:] if 'mc' in model_name else model_name[1:]
            model_tag = '_'.join([best_seed] + control[:-1] + ['best'])
            shutil.copy(os.path.join(backup_path, label, 'model', '{}.pt'.format(model_tag)),
                        os.path.join(model_path, '{}.pt'.format(model_tag)))
            if 'pixelcnn' in model_name:
                ae_tag = '_'.join([best_seed, control[0], control[1], cfg['ae_name'], 'best'])
                shutil.copy(os.path.join(backup_path, cfg['ae_name'], 'model', '{}.pt'.format(ae_tag)),
                            os.path.join(model_path, '{}.pt'.format(ae_tag)))
            script_name = '{}.py'.format(filename)
            control_name = '0.5' if 'mc' in model_name else None
            controls = [best_seed, data_name, model_name, control_name, save_per_mode]
            s.extend(['CUDA_VISIBLE_DEVICES="0" python {} --init_seed {} --data_name {} --model_name {} '
                      '--control_name {} --save_per_mode {}\n'.format(script_name, *controls)])
    else:
        for k, v in processed_result.items():
            vis(s, control + [k], v)
    return


if __name__ == '__main__':
    main()