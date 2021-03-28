import os
import itertools
import json
import numpy as np
import pandas as pd
import math
from utils import save, load, makedir_exist_ok
from config import cfg
import matplotlib.pyplot as plt
from collections import defaultdict

result_path = './output/result'
vis_path = './output/vis'
num_experiments = 1
exp = [str(x) for x in list(range(num_experiments))]


def make_controls(data_names, model_names, control_name):
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
    controls = exp + data_names + model_names + [control_names]
    controls = list(itertools.product(*controls))
    return controls


def make_control_list(file, model):
    model_names = [[model]]
    if file in ['interm']:
        if model in ['conv']:
            data_names = [['MNIST', 'CIFAR10']]
            control_name = [[['2', '4', '8'], [file], ['100'], ['none'], ['none'], ['none']]]
            control_2_4_8 = make_controls(data_names, model_names, control_name)
            data_names = [['ModelNet40']]
            control_name = [[['12'], [file], ['100'], ['none'], ['none'], ['none']]]
            control_12 = make_controls(data_names, model_names, control_name)
            controls = control_2_4_8 + control_12
        elif model in ['lstm']:
            data_names = [['MIMIC']]
            control_name = [[['2', '4', '8'], [file], ['100'], ['none'], ['none'], ['none']]]
            control_2_4_8 = make_controls(data_names, model_names, control_name)
            controls = control_2_4_8
        else:
            raise ValueError('Not valid model')
    elif file in ['late']:
        if model in ['linear']:
            data_names = [['Blob', 'Iris', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [[['2', '4'], [file], ['100'], ['none'], ['none'], ['none']]]
            control_2_4 = make_controls(data_names, model_names, control_name)
            data_names = [['Blob', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [[['8'], [file], ['100'], ['none'], ['none'], ['none']]]
            control_8 = make_controls(data_names, model_names, control_name)
            controls = control_2_4 + control_8
        elif model in ['conv']:
            data_names = [['MNIST', 'CIFAR10']]
            control_name = [[['2', '4', '8'], [file], ['100'], ['none'], ['none'], ['none']]]
            control_2_4_8 = make_controls(data_names, model_names, control_name)
            data_names = [['ModelNet40']]
            control_name = [[['12'], [file], ['100'], ['none'], ['none'], ['none']]]
            control_12 = make_controls(data_names, model_names, control_name)
            controls = control_2_4_8 + control_12
        elif model in ['lstm']:
            data_names = [['MIMIC']]
            control_name = [[['2', '4', '8'], [file], ['100'], ['none'], ['none'], ['none']]]
            control_2_4_8 = make_controls(data_names, model_names, control_name)
            controls = control_2_4_8
        else:
            raise ValueError('Not valid model')
    elif file == 'noise':
        if model in ['linear']:
            data_names = [['Blob', 'Iris', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [[['4'], ['bag', 'stack'], ['100'], ['10'], ['search'], ['1', '5']]]
            control_4 = make_controls(data_names, model_names, control_name)
            controls = control_4
        elif model in ['conv']:
            data_names = [['MNIST', 'CIFAR10']]
            control_name = [[['4'], ['bag', 'stack'], ['10'], ['10'], ['search'], ['1', '5']]]
            control_4 = make_controls(data_names, model_names, control_name)
            data_names = [['ModelNet40']]
            control_name = [[['12'], ['bag', 'stack'], ['10'], ['10'], ['search'], ['1', '5']]]
            control_12 = make_controls(data_names, model_names, control_name)
            controls = control_4 + control_12
        elif model in ['lstm']:
            data_names = [['MIMIC']]
            control_name = [[['4'], ['bag', 'stack'], ['10'], ['10'], ['search'], ['1', '5']]]
            control_4 = make_controls(data_names, model_names, control_name)
            controls = control_4
        else:
            raise ValueError('Not valid model')
    elif file == 'rate':
        model_names = [[model]]
        if model in ['linear']:
            data_names = [['Blob', 'Iris', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [[['4'], ['stack'], ['100'], ['10'], ['fix'], ['0']]]
            control_4 = make_controls(data_names, model_names, control_name)
            controls = control_4
        elif model in ['conv']:
            data_names = [['MNIST', 'CIFAR10']]
            control_name = [[['4'], ['stack'], ['10'], ['10'], ['fix'], ['0']]]
            control_4 = make_controls(data_names, model_names, control_name)
            data_names = [['ModelNet40']]
            control_name = [[['12'], ['stack'], ['10'], ['10'], ['fix'], ['0']]]
            control_12 = make_controls(data_names, model_names, control_name)
            controls = control_4 + control_12
        elif model in ['lstm']:
            data_names = [['MIMIC']]
            control_name = [[['4'], ['stack'], ['10'], ['10'], ['fix'], ['0']]]
            control_4 = make_controls(data_names, model_names, control_name)
            controls = control_4
        else:
            raise ValueError('Not valid model')
    elif file == 'assist':
        model_names = [[model]]
        if model in ['linear']:
            data_names = [['Blob', 'Iris', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [[['1'], ['none'], ['100'], ['10'], ['search'], ['0']]]
            control_1 = make_controls(data_names, model_names, control_name)
            data_names = [['Blob', 'Iris', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [[['2', '4'], ['none', 'bag', 'stack'], ['100'], ['10'], ['search'], ['0']]]
            control_2_4 = make_controls(data_names, model_names, control_name)
            data_names = [['Blob', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [[['8'], ['none', 'bag', 'stack'], ['100'], ['10'], ['search'], ['0']]]
            control_8 = make_controls(data_names, model_names, control_name)
            controls = control_1 + control_2_4 + control_8
        elif model in ['conv']:
            data_names = [['MNIST', 'CIFAR10']]
            control_name = [[['1'], ['none'], ['10'], ['10'], ['search'], ['0']]]
            control_1_1 = make_controls(data_names, model_names, control_name)
            data_names = [['MNIST', 'CIFAR10']]
            control_name = [[['2', '4', '8'], ['none', 'bag', 'stack'], ['10'], ['10'], ['search'], ['0']]]
            control_2_4_8 = make_controls(data_names, model_names, control_name)
            data_names = [['ModelNet40']]
            control_name = [[['1'], ['none'], ['10'], ['10'], ['search'], ['0']]]
            control_1_2 = make_controls(data_names, model_names, control_name)
            data_names = [['ModelNet40']]
            control_name = [[['12'], ['none', 'bag', 'stack'], ['10'], ['10'], ['search'], ['0']]]
            control_12 = make_controls(data_names, model_names, control_name)
            controls = control_1_1 + control_2_4_8 + control_1_2 + control_12
        elif model in ['lstm']:
            data_names = [['MIMIC']]
            control_name = [[['1'], ['none'], ['10'], ['10'], ['search'], ['0']]]
            control_1 = make_controls(data_names, model_names, control_name)
            data_names = [['MIMIC']]
            control_name = [[['2', '4', '8'], ['none', 'bag', 'stack'], ['10'], ['10'], ['search'], ['0']]]
            control_2_4_8 = make_controls(data_names, model_names, control_name)
            controls = control_1 + control_2_4_8
        else:
            raise ValueError('Not valid model')
    else:
        raise ValueError('Not valid file')
    return controls


def main():
    files = ['interm', 'late', 'noise', 'rate', 'assist']
    models = ['linear', 'conv', 'lstm']
    controls = []
    for file in files:
        for model in models:
            if file == 'interm' and model == 'linear':
                continue
            controls += make_control_list(file, model)
    processed_result_exp, processed_result_history = process_result(controls)
    with open('{}/processed_result_exp.json'.format(result_path), 'w') as fp:
        json.dump(processed_result_exp, fp, indent=2)
    save(processed_result_exp, os.path.join(result_path, 'processed_result_exp.pt'))
    save(processed_result_history, os.path.join(result_path, 'processed_result_history.pt'))
    extracted_processed_result_exp = {}
    extracted_processed_result_history = {}
    extract_processed_result(extracted_processed_result_exp, processed_result_exp, [])
    extract_processed_result(extracted_processed_result_history, processed_result_history, [])
    df_exp = make_df_exp(extracted_processed_result_exp)
    df_history = make_df_history(extracted_processed_result_history)
    make_vis(df_history)
    exit()
    return


def process_result(controls):
    processed_result_exp, processed_result_history = {}, {}
    for control in controls:
        model_tag = '_'.join(control)
        extract_result(list(control), model_tag, processed_result_exp, processed_result_history)
    summarize_result(processed_result_exp)
    summarize_result(processed_result_history)
    return processed_result_exp, processed_result_history


def extract_result(control, model_tag, processed_result_exp, processed_result_history):
    if len(control) == 1:
        exp_idx = exp.index(control[0])
        base_result_path_i = os.path.join(result_path, '{}.pt'.format(model_tag))
        if os.path.exists(base_result_path_i):
            base_result = load(base_result_path_i)
            for k in base_result['logger']['test'].history:
                metric_name = k.split('/')[1]
                if metric_name not in processed_result_exp:
                    processed_result_exp[metric_name] = {'exp': [None for _ in range(num_experiments)]}
                    processed_result_history[metric_name] = {'history': [None for _ in range(num_experiments)]}
                if metric_name in ['Loss', 'MAD']:
                    processed_result_exp[metric_name]['exp'][exp_idx] = min(base_result['logger']['test'].history[k])
                else:
                    processed_result_exp[metric_name]['exp'][exp_idx] = max(base_result['logger']['test'].history[k])
                processed_result_history[metric_name]['history'][exp_idx] = base_result['logger']['test'].history[k]
            if 'assist' in base_result:
                if 'Assist-Rate' not in processed_result_history:
                    processed_result_history['Assist-Rate'] = {'history': [None for _ in range(num_experiments)]}
                processed_result_history['Assist-Rate']['history'][exp_idx] = base_result['assist'].assist_rates[1:]
                if base_result['assist'].assist_parameters[1] is not None:
                    if 'Assist-Parameters' not in processed_result_history:
                        processed_result_history['Assist-Parameters'] = {
                            'history': [None for _ in range(num_experiments)]}
                    processed_result_history['Assist-Parameters']['history'][exp_idx] = [
                        base_result['assist'].assist_parameters[i]['stack'].softmax(dim=-1).numpy() for i in
                        range(1, len(base_result['assist'].assist_parameters))]
        else:
            print('Missing {}'.format(base_result_path_i))
    else:
        if control[1] not in processed_result_exp:
            processed_result_exp[control[1]] = {}
            processed_result_history[control[1]] = {}
        extract_result([control[0]] + control[2:], model_tag, processed_result_exp[control[1]],
                       processed_result_history[control[1]])
    return


def summarize_result(processed_result):
    if 'exp' in processed_result:
        pivot = 'exp'
        processed_result[pivot] = np.stack(processed_result[pivot], axis=0)
        processed_result['mean'] = np.mean(processed_result[pivot], axis=0).item()
        processed_result['std'] = np.std(processed_result[pivot], axis=0).item()
        processed_result['max'] = np.max(processed_result[pivot], axis=0).item()
        processed_result['min'] = np.min(processed_result[pivot], axis=0).item()
        processed_result['argmax'] = np.argmax(processed_result[pivot], axis=0).item()
        processed_result['argmin'] = np.argmin(processed_result[pivot], axis=0).item()
        processed_result[pivot] = processed_result[pivot].tolist()
    elif 'history' in processed_result:
        pivot = 'history'
        processed_result[pivot] = np.stack(processed_result[pivot], axis=0)
        processed_result['mean'] = np.mean(processed_result[pivot], axis=0)
        processed_result['std'] = np.std(processed_result[pivot], axis=0)
        processed_result['max'] = np.max(processed_result[pivot], axis=0)
        processed_result['min'] = np.min(processed_result[pivot], axis=0)
        processed_result['argmax'] = np.argmax(processed_result[pivot], axis=0)
        processed_result['argmin'] = np.argmin(processed_result[pivot], axis=0)
        processed_result[pivot] = processed_result[pivot].tolist()
    else:
        for k, v in processed_result.items():
            summarize_result(v)
        return
    return


def extract_processed_result(extracted_processed_result, processed_result, control):
    if 'exp' in processed_result or 'history' in processed_result:
        exp_name = '_'.join(control[:-1])
        metric_name = control[-1]
        if exp_name not in extracted_processed_result:
            extracted_processed_result[exp_name] = defaultdict()
        extracted_processed_result[exp_name]['{}_mean'.format(metric_name)] = processed_result['mean']
        extracted_processed_result[exp_name]['{}_std'.format(metric_name)] = processed_result['std']
    else:
        for k, v in processed_result.items():
            extract_processed_result(extracted_processed_result, v, control + [k])
    return


def write_xlsx(path, df, startrow=0):
    writer = pd.ExcelWriter(path, engine='xlsxwriter')
    for df_name in df:
        df[df_name] = pd.concat(df[df_name])
        df[df_name].to_excel(writer, sheet_name='Sheet1', startrow=startrow + 1)
        writer.sheets['Sheet1'].write_string(startrow, 0, df_name)
        startrow = startrow + len(df[df_name].index) + 3
    writer.save()
    return


def make_df_exp(extracted_processed_result_exp):
    df = defaultdict(list)
    for exp_name in extracted_processed_result_exp:
        control = exp_name.split('_')
        data_name, model_name, num_users, assist_mode, local_epoch, global_epoch, assist_rate_mode, noise = control
        index_name = ['_'.join([assist_mode, local_epoch, global_epoch, assist_rate_mode, noise])]
        df_name = '_'.join([data_name, model_name, num_users])
        df[df_name].append(pd.DataFrame(data=extracted_processed_result_exp[exp_name], index=index_name))
    write_xlsx('{}/result_exp.xlsx'.format(result_path), df)
    return df


def make_df_history(extracted_processed_result_history):
    df = defaultdict(list)
    for exp_name in extracted_processed_result_history:
        control = exp_name.split('_')
        data_name, model_name, num_users, assist_mode, local_epoch, global_epoch, assist_rate_mode, noise = control
        index_name = ['_'.join([assist_mode, local_epoch, global_epoch, assist_rate_mode, noise])]
        for k in extracted_processed_result_history[exp_name]:
            df_name = '_'.join([data_name, model_name, num_users, k])
            df[df_name].append(
                pd.DataFrame(data=extracted_processed_result_history[exp_name][k].reshape(1, -1), index=index_name))
    write_xlsx('{}/result_history.xlsx'.format(result_path), df)
    return df


def make_vis(df):
    color = {'Joint': 'red', 'Alone': 'orange', 'GAL-b': 'dodgerblue', 'GAL-s': 'green'}
    linestyle = {'Joint': '-', 'Alone': '--', 'GAL-b': ':', 'GAL-s': '-.'}
    marker = {'Joint': 's', 'Alone': '^', 'GAL-b': 'd', 'GAL-s': '*'}
    loc = {'Loss': 'upper right', 'Accuracy': 'lower right', 'MAD': 'upper right',
           'Gradient assisted learning rates': 'upper right', 'Assistance weights': 'upper right'}
    marker_noise_mp = {'1': 'v', '5': '^'}
    assist_mode_map = {'bag': 'GAL-b', 'stack': 'GAL-s', 'none': 'Alone'}
    color_ap = ['red', 'orange']
    linestyle_ap = ['-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--', ':', '-.']
    marker_ap = ['o', 'o', 'o', 'o', 's', 's', 's', 's', 'v', 'v', 'v', 'v', '^', '^', '^', '^']
    fontsize = {'legend': 16, 'label': 16, 'ticks': 16}
    capsize = 5
    save_format = 'pdf'
    markevery = 1
    fig = {}
    for df_name in df:
        data_name, model_name, num_users, metric_name, stat = df_name.split('_')
        if num_users == '1':
            continue
        if stat == 'std':
            continue
        df_name_std = '_'.join([data_name, model_name, num_users, metric_name, 'std'])
        if metric_name in ['Loss', 'Accuracy', 'MAD', 'Assist-Rate']:
            joint_df_name = '_'.join([data_name, model_name, '1', metric_name, stat])
            joint_df_name_df_name_std = '_'.join([data_name, model_name, '1', metric_name, 'std'])
            for ((index, row), (_, row_std)) in zip(df[joint_df_name].iterrows(),
                                                    df[joint_df_name_df_name_std].iterrows()):
                assist_mode, local_epoch, global_epoch, assist_rate_mode, noise = index.split('_')
                _metric_name = 'Gradient assisted learning rates' if metric_name == 'Assist-Rate' else metric_name
                xticks = np.arange(0, int(global_epoch) + 1, step=markevery)
                _assist_rate_mode = assist_rate_mode
                tag = 'assist'
                fig_name = '{}_{}'.format(df_name, tag)
                fig[fig_name] = plt.figure(fig_name)
                if metric_name in ['Loss', 'Accuracy', 'MAD']:
                    x = np.arange(0, int(global_epoch) + 1)
                else:
                    x = np.arange(1, int(global_epoch) + 1)
                y = row.to_numpy()
                label_name = 'Joint'
                index_name = 'Joint'
                plt.plot(x, y, color=color[index_name], linestyle=linestyle[index_name], label=label_name,
                         marker=marker[index_name], markevery=markevery)
                plt.legend(loc=loc[_metric_name], fontsize=fontsize['legend'])
                plt.xlabel('Assistance rounds', fontsize=fontsize['label'])
                plt.ylabel(_metric_name, fontsize=fontsize['label'])
                plt.xticks(xticks, fontsize=fontsize['ticks'])
                plt.yticks(fontsize=fontsize['ticks'])
            for ((index, row), (_, row_std)) in zip(df[df_name].iterrows(), df[df_name_std].iterrows()):
                assist_mode, local_epoch, global_epoch, assist_rate_mode, noise = index.split('_')
                if assist_mode in ['interm', 'late']:
                    continue
                _metric_name = 'Gradient assisted learning rates' if metric_name == 'Assist-Rate' else metric_name
                xticks = np.arange(0, int(global_epoch) + 1, step=markevery)
                _assist_mode = assist_mode_map[assist_mode]
                if int(noise) == 0:
                    tag = 'assist'
                    fig_name = '{}_{}'.format(df_name, tag)
                    label_name = '$M={}$, {}, {}'.format(num_users, _assist_mode, assist_rate_mode)
                else:
                    tag = 'noise'
                    fig_name = '{}_{}'.format(df_name, tag)
                    label_name = '$M={}$, {}, $\sigma={}$'.format(num_users, _assist_mode, noise)
                fig[fig_name] = plt.figure(fig_name)
                if metric_name in ['Loss', 'Accuracy', 'MAD']:
                    x = np.arange(0, int(global_epoch) + 1)
                else:
                    x = np.arange(1, int(global_epoch) + 1)
                y = row.to_numpy()
                yerr = row_std.to_numpy()
                index_name = _assist_mode
                if assist_rate_mode == 'fix':
                    _color = 'olive'
                else:
                    _color = color[index_name]
                if int(noise) == 0:
                    _marker = marker[index_name]
                else:
                    _marker = marker_noise_mp[noise]
                if _metric_name == 'Gradient assisted learning rates':
                    plt.plot(x, y, color=_color, linestyle=linestyle[index_name], label=label_name,
                             marker=_marker, markevery=markevery)
                else:
                    plt.errorbar(x, y, yerr=yerr, capsize=capsize, color=_color,
                                 linestyle=linestyle[index_name], label=label_name, marker=_marker,
                                 markevery=markevery)
                plt.legend(loc=loc[_metric_name], fontsize=fontsize['legend'])
                plt.xlabel('Assistance rounds', fontsize=fontsize['label'])
                plt.ylabel(_metric_name, fontsize=fontsize['label'])
                plt.xticks(xticks, fontsize=fontsize['ticks'])
                plt.yticks(fontsize=fontsize['ticks'])
        elif metric_name == 'Assist-Parameters':
            for ((index, row), (_, row_std)) in zip(df[df_name].iterrows(), df[df_name].iterrows()):
                assist_mode, local_epoch, global_epoch, assist_rate_mode, noise = index.split('_')
                x = np.arange(1, int(global_epoch) + 1)
                _metric_name = 'Assistance weights'
                xticks = np.arange(0, int(global_epoch) + 1, step=markevery)
                tag = '_'.join([assist_rate_mode, noise])
                fig_name = '{}_{}'.format(df_name, tag)
                for i in reversed(range(int(num_users))):
                    label_name = '$m={}$'.format(i + 1)
                    y = row.to_numpy().reshape(int(global_epoch), -1)[:, i]
                    fig[fig_name] = plt.figure(fig_name)
                    _color_ap = color_ap[int(i // (int(num_users) // 2))]
                    plt.plot(x, y, color=_color_ap, linestyle=linestyle_ap[i], label=label_name, marker=marker_ap[i],
                             markevery=1)
                    plt.legend(loc=loc[_metric_name], fontsize=fontsize['legend'])
                    plt.xlabel('Assistance rounds', fontsize=fontsize['label'])
                    plt.ylabel(_metric_name, fontsize=fontsize['label'])
                    plt.xticks(xticks, fontsize=fontsize['ticks'])
                    plt.yticks(fontsize=fontsize['ticks'])
        else:
            raise ValueError('Not valid metric name')
    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        plt.grid()
        fig_path = '{}/{}.{}'.format(vis_path, fig_name, save_format)
        makedir_exist_ok(vis_path)
        plt.savefig(fig_path, dpi=500, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return


if __name__ == '__main__':
    main()
