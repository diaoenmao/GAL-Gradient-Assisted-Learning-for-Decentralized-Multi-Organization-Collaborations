import os
import itertools
import json
import numpy as np
import pandas as pd
from utils import save, load, makedir_exist_ok
from config import cfg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict

result_path = './output/result'
vis_path = './output/vis'
num_experiments = 1
exp = [str(x) for x in list(range(num_experiments))]
# colors = cm.rainbow(np.linspace(1, 0, len(model_split_rate_key)))
# model_color = {model_split_rate_key[i]: colors[i] for i in range(len(model_split_rate_key))}
metric_name_dict = {'MNIST': 'Accuracy', 'CIFAR10': 'Accuracy', 'WikiText2': 'Perplexity'}
loc_dict = {'MNIST': 'lower right', 'CIFAR10': 'lower right', 'WikiText2': 'upper right'}
fontsize = 16


def make_controls(data_names, model_names, control_name):
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
    control_names = [control_names]
    controls = exp + data_names + model_names + control_names
    controls = list(itertools.product(*controls))
    return controls


def make_control_list(model_name):
    model_names = [[model_name]]
    if model_name in ['linear', 'mlp']:
        local_epoch = ['1', '10', '100']
        data_names = [['Blob', 'Iris', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR', 'MNIST', 'CIFAR10']]
        control_name = [[['1'], ['none'], local_epoch, ['10']]]
        control_1 = make_controls(data_names, model_names, control_name)
        data_names = [['Blob', 'Iris', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR', 'MNIST', 'CIFAR10']]
        control_name = [[['2', '4'], ['none', 'bag', 'stack'], local_epoch, ['10']]]
        control_2_4 = make_controls(data_names, model_names, control_name)
        data_names = [['Blob', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR', 'MNIST', 'CIFAR10']]
        control_name = [[['8'], ['none', 'bag', 'stack'], local_epoch, ['10']]]
        control_8 = make_controls(data_names, model_names, control_name)
        controls = control_1 + control_2_4 + control_8
    elif model_name in ['conv', 'resnet18']:
        local_epoch = ['1', '10']
        data_names = [['MNIST', 'CIFAR10']]
        control_name = [[['1'], ['none'], local_epoch, ['10']]]
        control_1 = make_controls(data_names, model_names, control_name)
        data_names = [['MNIST', 'CIFAR10']]
        control_name = [[['2', '4', '8'], ['none', 'bag', 'stack'], local_epoch, ['10']]]
        control_2_4_8 = make_controls(data_names, model_names, control_name)
        controls = control_1 + control_2_4_8
    elif model_name in ['conv-linear', 'resnet18-linear']:
        local_epoch = ['1', '10']
        data_names = [['MNIST', 'CIFAR10']]
        control_name = [[['1'], ['none'], local_epoch, ['50']]]
        control_1 = make_controls(data_names, model_names, control_name)
        data_names = [['MNIST', 'CIFAR10']]
        control_name = [[['2', '4', '8'], ['none', 'bag', 'stack'], local_epoch, ['50']]]
        control_2_4_8 = make_controls(data_names, model_names, control_name)
        controls = control_1 + control_2_4_8
    else:
        raise ValueError('Not valid model name')
    return controls


def main():
    linear_control_list = make_control_list('linear')
    mlp_control_list = make_control_list('mlp')
    conv_control_list = make_control_list('conv')
    resnet18_control_list = make_control_list('resnet18')
    convlinear_control_list = make_control_list('conv-linear')
    resnet18linear_control_list = make_control_list('resnet18-linear')
    controls = linear_control_list + mlp_control_list + conv_control_list + resnet18_control_list + \
               convlinear_control_list + resnet18linear_control_list
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
                if metric_name in ['Loss', 'RMSE']:
                    processed_result_exp[metric_name]['exp'][exp_idx] = min(base_result['logger']['test'].history[k])
                else:
                    processed_result_exp[metric_name]['exp'][exp_idx] = max(base_result['logger']['test'].history[k])
                processed_result_history[metric_name]['history'][exp_idx] = base_result['logger']['test'].history[k]
            if 'assist_rate' not in processed_result_history:
                processed_result_history['assist_rate'] = {'history': [None for _ in range(num_experiments)]}
            processed_result_history['assist_rate']['history'][exp_idx] = base_result['assist'].assist_rates[0][1:]
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


def make_df_exp(extracted_processed_result_exp):
    df = defaultdict(list)
    for exp_name in extracted_processed_result_exp:
        control = exp_name.split('_')
        data_name, model_name, num_users, assist_mode, local_epoch, global_epoch = control
        index_name = ['_'.join([num_users, assist_mode, local_epoch, global_epoch])]
        df_name = '_'.join([data_name, model_name])
        df[df_name].append(pd.DataFrame(data=extracted_processed_result_exp[exp_name], index=index_name))
    startrow = 0
    writer = pd.ExcelWriter('{}/result_exp.xlsx'.format(result_path), engine='xlsxwriter')
    for df_name in df:
        df[df_name] = pd.concat(df[df_name])
        df[df_name].to_excel(writer, sheet_name='Sheet1', startrow=startrow + 1)
        writer.sheets['Sheet1'].write_string(startrow, 0, df_name)
        startrow = startrow + len(df[df_name].index) + 3
    writer.save()
    return df


def make_df_history(extracted_processed_result_history):
    df = defaultdict(list)
    for exp_name in extracted_processed_result_history:
        control = exp_name.split('_')
        data_name, model_name, num_users, assist_mode, local_epoch, global_epoch = control
        index_name = ['_'.join([num_users, assist_mode, local_epoch, global_epoch])]
        df_name_loss = '_'.join([data_name, model_name, 'Loss'])
        df[df_name_loss].append(
            pd.DataFrame(data=extracted_processed_result_history[exp_name]['Loss_mean'].reshape(1, -1),
                         index=index_name))
        if 'Accuracy_mean' in extracted_processed_result_history[exp_name]:
            df_name_acc = '_'.join([data_name, model_name, 'Accuracy'])
            df[df_name_acc].append(
                pd.DataFrame(data=extracted_processed_result_history[exp_name]['Accuracy_mean'].reshape(1, -1),
                             index=index_name))
        if 'RMSE_mean' in extracted_processed_result_history[exp_name]:
            df_name_acc = '_'.join([data_name, model_name, 'RMSE'])
            df[df_name_acc].append(
                pd.DataFrame(data=extracted_processed_result_history[exp_name]['RMSE_mean'].reshape(1, -1),
                             index=index_name))
        if 'assist_rate_mean' in extracted_processed_result_history[exp_name]:
            df_name_assist_rate = '_'.join([data_name, model_name, 'assist_rate'])
            df[df_name_assist_rate].append(
                pd.DataFrame(data=extracted_processed_result_history[exp_name]['assist_rate_mean'].reshape(1, -1),
                             index=index_name))
    startrow = 0
    writer = pd.ExcelWriter('{}/result_history.xlsx'.format(result_path), engine='xlsxwriter')
    for df_name in df:
        df[df_name] = pd.concat(df[df_name])
        df[df_name].to_excel(writer, sheet_name='Sheet1', startrow=startrow + 1)
        writer.sheets['Sheet1'].write_string(startrow, 0, df_name)
        startrow = startrow + len(df[df_name].index) + 3
    writer.save()
    return df


def make_vis(df):
    fig = {}
    for df_name in df:
        if 'fix' in df_name and 'none' not in df_name:
            control = df_name.split('_')
            data_name = control[0]
            metric_name = metric_name_dict[data_name]
            label_name = control[-1]
            x = df[df_name]['Params_mean']
            if 'non-iid-2' in df_name:
                fig_name = '{}_{}_local'.format('_'.join(control[:-1]), label_name[0])
                fig[fig_name] = plt.figure(fig_name)
                y = df[df_name]['Local-{}_mean'.format(metric_name)]
                plt.plot(x, y, '^--', label=label_name)
                plt.legend(loc=loc_dict[data_name], fontsize=fontsize)
                plt.xlabel('Number of Model Parameters', fontsize=fontsize)
                plt.ylabel(metric_name, fontsize=fontsize)
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)
                plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
                fig_name = '{}_{}_global'.format('_'.join(control[:-1]), label_name[0])
                fig[fig_name] = plt.figure(fig_name)
                y = df[df_name]['Global-{}_mean'.format(metric_name)]
                plt.plot(x, y, '^--', label=label_name)
                plt.legend(loc=loc_dict[data_name], fontsize=fontsize)
                plt.xlabel('Number of Model Parameters', fontsize=fontsize)
                plt.ylabel(metric_name, fontsize=fontsize)
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)
                plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            elif 'iid' in df_name:
                fig_name = '{}_{}'.format('_'.join(control[:-1]), label_name[0])
                fig[fig_name] = plt.figure(fig_name)
                y = df[df_name]['Global-{}_mean'.format(metric_name)]
                plt.plot(x, y, '^--', label=label_name)
                plt.legend(loc=loc_dict[data_name], fontsize=fontsize)
                plt.xlabel('Number of Model Parameters', fontsize=fontsize)
                plt.ylabel(metric_name, fontsize=fontsize)
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)
                plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            else:
                raise ValueError('Not valid df name')
    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        plt.grid()
        fig_path = '{}/{}.{}'.format(vis_path, fig_name, cfg['save_format'])
        makedir_exist_ok(vis_path)
        plt.savefig(fig_path, dpi=500, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return


if __name__ == '__main__':
    main()
