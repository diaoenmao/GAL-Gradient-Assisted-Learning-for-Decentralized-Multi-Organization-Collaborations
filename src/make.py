import argparse
import itertools

parser = argparse.ArgumentParser(description='Config')
parser.add_argument('--run', default=None, type=str)
parser.add_argument('--model', default=None, type=str)
parser.add_argument('--file', default=None, type=str)
parser.add_argument('--num_gpus', default=4, type=int)
parser.add_argument('--world_size', default=1, type=int)
parser.add_argument('--init_seed', default=0, type=int)
parser.add_argument('--round', default=4, type=int)
parser.add_argument('--experiment_step', default=1, type=int)
parser.add_argument('--num_experiments', default=1, type=int)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--resume_mode', default=0, type=int)
args = vars(parser.parse_args())


def main():
    run = args['run']
    model = args['model']
    file = args['file']
    num_gpus = args['num_gpus']
    world_size = args['world_size']
    round = args['round']
    experiment_step = args['experiment_step']
    init_seed = args['init_seed']
    num_experiments = args['num_experiments']
    num_epochs = args['num_epochs']
    resume_mode = args['resume_mode']
    gpu_ids = [','.join(str(i) for i in list(range(x, x + world_size))) for x in list(range(0, num_gpus, world_size))]
    if file is None:
        if model in ['resnet18']:
            file = 'classifier'
            control_names = [['None']]
            script_name = [['{}_{}.py'.format(run, file)]]
            filename = '{}_{}'.format(run, model)
        elif model in ['scgan']:
            file = 'gan'
            control_names = [['None']]
            script_name = [['{}_{}.py'.format(run, file)]]
            filename = '{}_{}'.format(run, model)
        else:
            raise ValueError('Not valid model')
    else:
        control_names = [['None']]
        script_name = [['{}_{}.py'.format(run, file)]] if run is not None else [['{}.py'.format(file)]]
        filename = '{}_{}'.format(run, file) if run is not None else file
    data_names = [['MNIST', 'SVHN', 'CIFAR10', 'Flower', 'Omniglot']]
    model_names = [[model]]
    init_seeds = [list(range(init_seed, init_seed + num_experiments, experiment_step))]
    num_experiments = [[experiment_step]]
    num_epochs = [[num_epochs]]
    resume_mode = [[resume_mode]]
    world_size = [[world_size]]
    s = '#!/bin/bash\n'
    k = 0
    controls = script_name + data_names + model_names + init_seeds + world_size + num_experiments + \
               num_epochs + resume_mode + control_names
    controls = list(itertools.product(*controls))
    for i in range(len(controls)):
        controls[i] = list(controls[i])
        s = s + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --data_name {} --model_name {} --init_seed {} ' \
                '--world_size {} --num_experiments {} --num_epochs {} --resume_mode {} --control_name {}&\n'.format(
            gpu_ids[k % len(gpu_ids)], *controls[i])
        if k % round == round - 1:
            s = s[:-2] + '\nwait\n'
        k = k + 1
    print(s)
    run_file = open('./{}.sh'.format(filename), 'w')
    run_file.write(s)
    run_file.close()
    return


if __name__ == '__main__':
    main()