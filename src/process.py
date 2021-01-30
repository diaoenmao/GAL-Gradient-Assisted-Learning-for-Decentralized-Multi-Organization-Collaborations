import os
import itertools
import json
import numpy as np
import pandas as pd
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
    control_names = [control_names]
    controls = exp + data_names + model_names + control_names
    controls = list(itertools.product(*controls))
    return controls


def make_control_list(model_name):
    model_names = [[model_name]]
    if model_name in ['linear', 'mlp']:
        local_epoch = ['10']
        data_names = [['Blob', 'Iris', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
        control_name = [[['1'], ['none'], local_epoch, ['10']]]
        control_1 = make_controls(data_names, model_names, control_name)
        data_names = [['Blob', 'Iris', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
        control_name = [[['2', '4'], ['none', 'bag', 'stack'], local_epoch, ['10']]]
        control_2_4 = make_controls(data_names, model_names, control_name)
        data_names = [['Blob', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
        control_name = [[['8'], ['none', 'bag', 'stack'], local_epoch, ['10']]]
        control_8 = make_controls(data_names, model_names, control_name)
        controls = control_1 + control_2_4 + control_8
    elif model_name in ['conv']:
        local_epoch = ['10']
        data_names = [['MNIST']]
        control_name = [[['1'], ['none'], local_epoch, ['10']]]
        control_1 = make_controls(data_names, model_names, control_name)
        data_names = [['MNIST']]
        control_name = [[['2', '4', '8'], ['none', 'bag', 'stack'], local_epoch, ['10']]]
        control_2_4_8 = make_controls(data_names, model_names, control_name)
        controls = control_1 + control_2_4_8
    elif model_name in ['resnet18']:
        local_epoch = ['10']
        data_names = [['CIFAR10']]
        control_name = [[['1'], ['none'], local_epoch, ['10']]]
        control_1 = make_controls(data_names, model_names, control_name)
        data_names = [['CIFAR10']]
        control_name = [[['2', '4', '8'], ['none', 'bag', 'stack'], local_epoch, ['10']]]
        control_2_4_8 = make_controls(data_names, model_names, control_name)
        controls = control_1 + control_2_4_8
    elif model_name in ['conv-linear']:
        local_epoch = ['10']
        data_names = [['MNIST']]
        control_name = [[['1'], ['none'], local_epoch, ['10']]]
        control_1 = make_controls(data_names, model_names, control_name)
        data_names = [['MNIST']]
        control_name = [[['2', '4', '8'], ['none', 'bag', 'stack'], local_epoch, ['50']]]
        control_2_4_8 = make_controls(data_names, model_names, control_name)
        controls = control_1 + control_2_4_8
    elif model_name in ['resnet18-linear']:
        local_epoch = ['10']
        data_names = [['CIFAR10']]
        control_name = [[['1'], ['none'], local_epoch, ['10']]]
        control_1 = make_controls(data_names, model_names, control_name)
        data_names = [['CIFAR10']]
        control_name = [[['2', '4', '8'], ['none', 'bag', 'stack'], local_epoch, ['50']]]
        control_2_4_8 = make_controls(data_names, model_names, control_name)
        controls = control_1 + control_2_4_8
    else:
        raise ValueError('Not valid model name')
    return controls


def main():
    linear_control_list = make_control_list('linear')
    conv_control_list = make_control_list('conv')
    resnet18_control_list = make_control_list('resnet18')
    controls = linear_control_list + conv_control_list + resnet18_control_list
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
            if 'Assist-Rate' not in processed_result_history:
                processed_result_history['Assist-Rate'] = {'history': [None for _ in range(num_experiments)]}
            processed_result_history['Assist-Rate']['history'][exp_idx] = base_result['assist'].assist_rates[1:]
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
        index_name = ['_'.join([local_epoch, assist_mode])]
        df_name = '_'.join([data_name, model_name, num_users, global_epoch])
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
        index_name = ['_'.join([local_epoch, assist_mode])]
        df_name_loss = '_'.join([data_name, model_name, num_users, global_epoch, 'Loss'])
        df[df_name_loss].append(
            pd.DataFrame(data=extracted_processed_result_history[exp_name]['Loss_mean'].reshape(1, -1),
                         index=index_name))
        if 'Accuracy_mean' in extracted_processed_result_history[exp_name]:
            df_name_acc = '_'.join([data_name, model_name, num_users, global_epoch, 'Accuracy'])
            df[df_name_acc].append(
                pd.DataFrame(data=extracted_processed_result_history[exp_name]['Accuracy_mean'].reshape(1, -1),
                             index=index_name))
        if 'RMSE_mean' in extracted_processed_result_history[exp_name]:
            df_name_rmse = '_'.join([data_name, model_name, num_users, global_epoch, 'RMSE'])
            df[df_name_rmse].append(
                pd.DataFrame(data=extracted_processed_result_history[exp_name]['RMSE_mean'].reshape(1, -1),
                             index=index_name))
        if 'Assist-Rate_mean' in extracted_processed_result_history[exp_name]:
            df_name_assist_rate = '_'.join([data_name, model_name, num_users, global_epoch, 'Assist-Rate'])
            df[df_name_assist_rate].append(
                pd.DataFrame(data=extracted_processed_result_history[exp_name]['Assist-Rate_mean'].reshape(1, -1),
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
    color_dict = {'Joint': 'red', 'Separate': 'orange', 'Bag': 'dodgerblue', 'Stack': 'green'}
    linestyle = {'1': '--', '10': '-', '100': ':'}
    marker_dict = {'Joint': {'1': 'o', '10': 's', '100': 'D'}, 'Separate': {'1': 'v', '10': '^', '100': '>'},
                   'Bag': {'1': 'p', '10': 'd', '100': 'h'}, 'Stack': {'1': 'X', '10': '*', '100': 'x'}}
    loc_dict = {'Loss': 'lower right', 'Accuracy': 'lower right', 'RMSE': 'lower right', 'Assist Rate': 'lower right'}
    fontsize = {'legend': 16, 'label': 16, 'ticks': 16}
    save_format = 'png'
    fig = {}
    for df_name in df:
        print(df_name)
        data_name, model_name, num_users, global_epoch, metric_name = df_name.split('_')
        if num_users == '1':
            continue
        baseline_df_name = '_'.join([data_name, model_name, '1', global_epoch, metric_name])
        if metric_name in ['Loss', 'Accuracy', 'RMSE']:
            x = np.arange(0, int(global_epoch) + 1)
        elif metric_name in ['Assist-Rate']:
            x = np.arange(1, int(global_epoch) + 1)
            metric_name = 'Assist Rate'
        else:
            raise ValueError('Not valid metric name')
        if global_epoch == '10':
            markevery = 1
            xticks = np.arange(0, int(global_epoch) + 1, step=markevery)
        elif global_epoch == '50':
            markevery = 10
            xticks = np.arange(int(global_epoch) + 1, step=markevery)
            xticks[0] = 1
        else:
            raise ValueError('Not valid global epoch')
        for index, row in df[baseline_df_name].iterrows():
            local_epoch, assist_mode = index.split('_')
            if assist_mode == 'none':
                assist_mode = 'Joint'
            else:
                raise ValueError('Not valid assist_mode')
            label_name = '{}'.format(assist_mode)
            y = row.to_numpy()
            fig[df_name] = plt.figure(df_name)
            plt.plot(x, y, color=color_dict[assist_mode], linestyle=linestyle[local_epoch], label=label_name,
                     marker=marker_dict[assist_mode][local_epoch], markevery=markevery)
            plt.legend(loc=loc_dict[metric_name], fontsize=fontsize['legend'])
            plt.xlabel('Assist Round (T)', fontsize=fontsize['label'])
            plt.ylabel(metric_name, fontsize=fontsize['label'])
            plt.xticks(xticks, fontsize=fontsize['ticks'])
            plt.yticks(fontsize=fontsize['ticks'])
        for index, row in df[df_name].iterrows():
            local_epoch, assist_mode = index.split('_')
            if assist_mode == 'none':
                assist_mode = 'Separate'
            elif assist_mode == 'bag':
                assist_mode = 'Bag'
            elif assist_mode == 'stack':
                assist_mode = 'Stack'
            else:
                raise ValueError('Not valid assist_mode')
            label_name = 'M={}, {}'.format(num_users, assist_mode)
            y = row.to_numpy()
            fig[df_name] = plt.figure(df_name)
            plt.plot(x, y, color=color_dict[assist_mode], linestyle=linestyle[local_epoch], label=label_name,
                     marker=marker_dict[assist_mode][local_epoch], markevery=markevery)
            plt.legend(loc=loc_dict[metric_name], fontsize=fontsize['legend'])
            plt.xlabel('Assist Round', fontsize=fontsize['label'])
            plt.ylabel(metric_name, fontsize=fontsize['label'])
            plt.xticks(xticks, fontsize=fontsize['ticks'])
            plt.yticks(fontsize=fontsize['ticks'])
        plt.grid()
        fig_path = '{}/{}.{}'.format(vis_path, df_name, save_format)
        makedir_exist_ok(vis_path)
        plt.savefig(fig_path, dpi=500, bbox_inches='tight', pad_inches=0)
        plt.close(df_name)
    return


if __name__ == '__main__':
    main()
