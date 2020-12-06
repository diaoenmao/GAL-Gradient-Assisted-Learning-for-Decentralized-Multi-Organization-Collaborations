import copy
import torch
import models
from config import cfg
from data import make_data_loader
from organization import Organization
from utils import make_optimizer, make_scheduler, collate, to_device


class Assist:
    def __init__(self, feature_split, assist_rate):
        self.feature_split = feature_split
        self.assist_rate = assist_rate
        self.model_name = self.make_model_name()
        self.reset()

    def reset(self):
        self.organization_outputs = [{split: None for split in cfg['data_size']} for _ in
                                     range(len(self.feature_split))]
        self.assist_parameters = [[None for _ in range(cfg['global']['num_epochs'])] for _ in
                                  range(len(self.feature_split))]
        return

    def make_model_name(self):
        model_name = cfg['model_name'].split('-')
        model_idx = torch.arange(len(model_name)).repeat(round(len(self.feature_split) / len(model_name)))
        model_idx = model_idx.tolist()[:len(self.feature_split)]
        model_name = [model_name[i] for i in model_idx]
        return model_name

    def make_data_loader(self, dataset):
        data_loader = [None for _ in range(len(self.feature_split))]
        for i in range(len(self.feature_split)):
            data_loader[i] = make_data_loader(dataset, self.model_name[i])
        return data_loader

    def make_organization(self):
        feature_split = self.feature_split
        model_name = self.model_name
        organization = [None for _ in range(len(feature_split))]
        for i in range(len(feature_split)):
            model_name_i = model_name[i]
            feature_split_i = feature_split[i]
            organization[i] = Organization(i, feature_split_i, model_name_i)
        return organization

    def update(self, iter, data_loader, new_organization_outputs):
        if cfg['assist_mode'] == 'none':
            for i in range(len(self.organization_outputs)):
                for split in self.organization_outputs[i]:
                    if self.organization_outputs[i][split] is None:
                        self.organization_outputs[i][split] = new_organization_outputs[i][split]
                    else:
                        self.organization_outputs[i][split]['target'] = self.organization_outputs[i][split]['target'] \
                                                                        - self.assist_rate * \
                                                                        new_organization_outputs[i][split]['target']
        elif cfg['assist_mode'] == 'bagging':
            organization_outputs = {split: {'id': torch.arange(cfg['data_size'][split]),
                                            'target': torch.zeros(cfg['data_size'][split], cfg['target_size'])}
                                    for split in cfg['data_size']}
            mask = {split: torch.zeros(cfg['data_size'][split], cfg['target_size']) for split in cfg['data_size']}
            for split in cfg['data_size']:
                for i in range(len(new_organization_outputs)):
                    id = new_organization_outputs[i][split]['id']
                    mask[split][id] += 1
                    organization_outputs[split]['target'][id] += new_organization_outputs[i][split]['target']
                organization_outputs[split]['target'] = organization_outputs[split]['target'] / mask[split]
            for i in range(len(self.organization_outputs)):
                for split in self.organization_outputs[i]:
                    if self.organization_outputs[i][split] is None:
                        self.organization_outputs[i][split] = copy.deepcopy(organization_outputs[split])
                    else:
                        self.organization_outputs[i][split]['target'] = self.organization_outputs[i][split]['target'] \
                                                                        - self.assist_rate * \
                                                                        organization_outputs[split]['target']
        elif cfg['assist_mode'] == 'stacking':
            # organization_outputs = {split: {'id': torch.arange(cfg['data_size'][split]),
            #                                 'target': torch.zeros(cfg['data_size'][split], cfg['target_size'])}
            #                         for split in cfg['data_size']}
            # # mask = {split: torch.zeros(cfg['data_size'][split], cfg['target_size']) for split in cfg['data_size']}
            # for split in cfg['data_size']:
            #     # for i in range(len(new_organization_outputs)):
            #     #     print(i)
            #     #     id = new_organization_outputs[i][split]['id']
            #     #     mask[split][id] += 1
            #     #     organization_outputs[split]['target'][id] += new_organization_outputs[i][split]['target']
            #     # organization_outputs[split]['target'] = organization_outputs[split]['target'] / mask[split]
            #     output = []
            #     for j in range(len(new_organization_outputs)):
            #         output.append(new_organization_outputs[j][split]['target'])
            #     organization_outputs[split]['target'] = torch.stack(output, dim=-1).mean(-1)
                # print(organization_outputs[split]['target'][:10])
            # for i in range(len(self.organization_outputs)):
            #     for split in self.organization_outputs[i]:
            #         if self.organization_outputs[i][split] is None:
            #             self.organization_outputs[i][split] = organization_outputs[split]
            #         else:
            #             self.organization_outputs[i][split]['target'] = self.organization_outputs[i][split]['target'] \
            #                                                             - self.assist_rate * \
            #                                                             organization_outputs[split]['target']
            print(new_organization_outputs[0]['train']['target'][:5])
            print(new_organization_outputs[0]['test']['target'][:5])
            _dataset = [{split: None for split in cfg['data_size']} for _ in range(len(self.feature_split))]
            for i in range(len(self.feature_split)):
                for split in data_loader[i]:
                    output = []
                    for j in range(len(new_organization_outputs)):
                        output.append(new_organization_outputs[j][split]['target'])
                    output = torch.stack(output, dim=-1)
                    if self.organization_outputs[i][split] is None:
                        _dataset[i][split] = torch.utils.data.TensorDataset(
                            torch.tensor(data_loader[i][split].dataset.id), output,
                            torch.tensor(data_loader[i][split].dataset.target))
                    else:
                        _dataset[i][split] = torch.utils.data.TensorDataset(
                            torch.tensor(data_loader[i][split].dataset.id),
                             output, torch.tensor(data_loader[i][split].dataset.target),
                            self.organization_outputs[i][split]['target'])
            # for i in range(len(self.feature_split)):
            #     _data_loader = make_data_loader(_dataset[i], 'assist')
            #     if 'train' in _data_loader:
            #         model = models.stack().to(cfg['device'])
            #         if iter > 0:
            #             model.load_state_dict(self.assist_parameters[i][iter - 1])
            #         model.train(True)
            #         optimizer = make_optimizer(model, 'assist')
            #         scheduler = make_scheduler(optimizer, 'assist')
            #         for assist_epoch in range(1, cfg['assist']['num_epochs'] + 1):
            #             for j, input in enumerate(_data_loader['train']):
            #                 if len(input) == 3:
            #                     input = {'id': input[0], 'output': input[1], 'target': input[2], 'assist': None}
            #                 else:
            #                     input = {'id': input[0], 'output': input[1], 'target': input[2], 'assist': input[3]}
            #                 input = to_device(input, cfg['device'])
            #                 optimizer.zero_grad()
            #                 output = model(input)
            #                 # output['loss'].backward()
            #                 # optimizer.step()
            #             scheduler.step()
            #         self.assist_parameters[i][iter] = model.to('cpu').state_dict()
            with torch.no_grad():

                # for split in data_loader[i]:
                #     # organization_outputs = {'id': torch.arange(cfg['data_size'][split]),
                #     #                         'target': torch.zeros(cfg['data_size'][split], cfg['target_size'])}
                #     model = models.stack().to(cfg['device'])
                #     model.load_state_dict(self.assist_parameters[i][iter])
                #     model.train(False)
                #     for j, input in enumerate(_data_loader[split]):
                #         if len(input) == 3:
                #             input = {'id': input[0], 'output': input[1], 'target': input[2], 'assist': None}
                #         else:
                #             input = {'id': input[0], 'output': input[1], 'target': input[2], 'assist': input[3]}
                #         input = to_device(input, cfg['device'])
                #         output = model(input)
                #         print(output['target'][:10], input['id'][:10])
                #         organization_outputs['target'][input['id']] = output['target'][input['id']].cpu()
                #     if self.organization_outputs[i][split] is None:
                #         self.organization_outputs[i][split] = organization_outputs
                #     else:
                #         self.organization_outputs[i][split]['target'] = self.organization_outputs[i][split][
                #                                                             'target'] - cfg['assist_rate'] * \
                #                                                         organization_outputs['target']
                # for i in range(len(self.organization_outputs)):
                #     organization_outputs = {split: {'id': torch.arange(cfg['data_size'][split]),
                #                                     'target': torch.zeros(cfg['data_size'][split],
                #                                                           cfg['target_size'])}
                #                             for split in cfg['data_size']}
                #     for split in self.organization_outputs[i]:
                #         model = models.stack().to(cfg['device'])
                #         # model.load_state_dict(self.assist_parameters[i][iter])
                #         model.train(False)
                #         output = []
                #         for j, input in enumerate(_data_loader[split]):
                #             if len(input) == 3:
                #                 input = {'id': input[0], 'output': input[1], 'target': input[2], 'assist': None}
                #             else:
                #                 input = {'id': input[0], 'output': input[1], 'target': input[2], 'assist': input[3]}
                #             input = to_device(input, cfg['device'])
                #             output.append(model(input)['target'].cpu())
                #         organization_outputs[split]['target'] = torch.cat(output, dim=0)
                #         # print(organization_outputs[split]['target'][:10])
                #         if self.organization_outputs[i][split] is None:
                #             self.organization_outputs[i][split] = organization_outputs[split]
                #         else:
                #             self.organization_outputs[i][split]['target'] = self.organization_outputs[i][split]['target'] \
                #                                                             - self.assist_rate * \
                #                                                             organization_outputs[split]['target']

                # organization_outputs = [{split: {'id': torch.arange(cfg['data_size'][split]),
                #                                 'target': torch.zeros(cfg['data_size'][split],
                #                                                       cfg['target_size'])}
                #                         for split in cfg['data_size']} for _ in range(len(self.organization_outputs))]
                # for i in range(len(self.organization_outputs)):
                #     _data_loader = make_data_loader(_dataset[i], 'assist')
                #     for split in cfg['data_size']:
                #         model = models.stack().to(cfg['device'])
                #         # model.load_state_dict(self.assist_parameters[i][iter])
                #         model.train(False)
                #         output = []
                #         for j, input in enumerate(_data_loader[split]):
                #             if len(input) == 3:
                #                 input = {'id': input[0], 'output': input[1], 'target': input[2], 'assist': None}
                #             else:
                #                 input = {'id': input[0], 'output': input[1], 'target': input[2], 'assist': input[3]}
                #             input = to_device(input, cfg['device'])
                #             output = model(input)
                #             organization_outputs[i][split]['target'][input['id']] = output['target'][input['id']].cpu()
                # print('kkkk')
                # print(organization_outputs[0]['train']['target'][:5])
                # print(organization_outputs[0]['test']['target'][:5])
                # print(organization_outputs[1]['train']['target'][:5])
                # print(organization_outputs[1]['test']['target'][:5])
                # print(organization_outputs[2]['train']['target'][:5])
                # print(organization_outputs[2]['test']['target'][:5])
                # print(organization_outputs[3]['train']['target'][:5])
                # print(organization_outputs[3]['test']['target'][:5])
                # print('kkkk')
                # for i in range(len(self.organization_outputs)):
                #     for split in self.organization_outputs[i]:
                #         if self.organization_outputs[i][split] is None:
                #             self.organization_outputs[i][split] = organization_outputs[i][split]
                #         else:
                #             self.organization_outputs[i][split]['target'] = self.organization_outputs[i][split]['target'] \
                #                                                             - self.assist_rate * \
                #                                                             organization_outputs[i][split]['target']

                organization_outputs = {split: {'id': torch.arange(cfg['data_size'][split]),
                                                'target': torch.zeros(cfg['data_size'][split],
                                                                      cfg['target_size'])}
                                        for split in cfg['data_size']}
                for i in range(len(self.organization_outputs)):
                    _data_loader = make_data_loader(_dataset[i], 'assist')
                    for split in cfg['data_size']:
                        model = models.stack().to(cfg['device'])
                        # model.load_state_dict(self.assist_parameters[i][iter])
                        model.train(False)
                        output = []
                        for j, input in enumerate(_data_loader[split]):
                            if len(input) == 3:
                                input = {'id': input[0], 'output': input[1], 'target': input[2], 'assist': None}
                            else:
                                input = {'id': input[0], 'output': input[1], 'target': input[2], 'assist': input[3]}
                            input = to_device(input, cfg['device'])
                            output = model(input)
                            organization_outputs[split]['target'][input['id']] = output['target'][input['id']].cpu()
                print('kkkk')
                print(organization_outputs['train']['target'][:5])
                print(organization_outputs['test']['target'][:5])
                print('kkkk')

                for i in range(len(self.organization_outputs)):
                    for split in self.organization_outputs[i]:
                        if self.organization_outputs[i][split] is None:
                            self.organization_outputs[i][split] = copy.deepcopy(organization_outputs[split])
                        else:
                            self.organization_outputs[i][split]['target'] = self.organization_outputs[i][split]['target'] \
                                                                            - self.assist_rate * \
                                                                            organization_outputs[split]['target']
                print(self.organization_outputs[0]['train']['target'][:5])
                print(self.organization_outputs[0]['test']['target'][:5])
                            # exit()



        elif cfg['assist_mode'] == 'attention':
            _dataset = [{split: None for split in cfg['data_size']} for _ in range(len(self.feature_split))]
            for i in range(len(self.organization_outputs)):
                for split in data_loader[i]:
                    assist = self.organization_outputs[i][split]
                    output = []
                    for j in range(len(new_organization_outputs)):
                        output.append(new_organization_outputs[j][split]['target'])
                    output = torch.stack(output, dim=-1)
                    if assist is None:
                        _dataset[i][split] = torch.utils.data.TensorDataset(
                            torch.tensor(data_loader[i][split].dataset.id), output,
                            torch.tensor(data_loader[i][split].dataset.target))
                    else:
                        assist = assist['target']
                        _dataset[i][split] = torch.utils.data.TensorDataset(
                            torch.tensor(data_loader[i][split].dataset.id),
                            assist, output, torch.tensor(data_loader[i][split].dataset.target))
            for i in range(len(self.organization_outputs)):
                _data_loader = make_data_loader(_dataset[i], 'assist')
                if 'train' in _data_loader:
                    model = models.attention().to(cfg['device'])
                    if iter > 0:
                        model.load_state_dict(self.assist_parameters[i][iter - 1])
                    model.train(True)
                    optimizer = make_optimizer(model, 'assist')
                    scheduler = make_scheduler(optimizer, 'assist')
                    for assist_epoch in range(1, cfg['assist']['num_epochs'] + 1):
                        for j, input in enumerate(_data_loader['train']):
                            if len(input) == 3:
                                input = {'id': input[0], 'assist': None, 'output': input[1], 'target': input[2]}
                            else:
                                input = {'id': input[0], 'assist': input[1], 'output': input[2], 'target': input[3]}
                            input = to_device(input, cfg['device'])
                            optimizer.zero_grad()
                            output = model(input)
                            output['loss'].backward()
                            optimizer.step()
                        scheduler.step()
                    self.assist_parameters[i][iter] = model.to('cpu').state_dict()
                with torch.no_grad():
                    for split in _data_loader:
                        organization_outputs = {'id': torch.arange(cfg['data_size'][split]),
                                                'target': torch.zeros(cfg['data_size'][split], cfg['target_size'])}
                        model = models.attention().to(cfg['device'])
                        model.load_state_dict(self.assist_parameters[i][iter])
                        model.train(False)
                        for j, input in enumerate(_data_loader[split]):
                            if len(input) == 3:
                                input = {'id': input[0], 'assist': None, 'output': input[1]}
                            else:
                                input = {'id': input[0], 'assist': input[1], 'output': input[2]}
                            input = to_device(input, cfg['device'])
                            output = model(input)
                            organization_outputs['target'][input['id']] = output['target'].cpu()
                        if self.organization_outputs[i][split] is None:
                            self.organization_outputs[i][split] = organization_outputs
                        else:
                            self.organization_outputs[i][split]['target'] = self.organization_outputs[i][split][
                                                                                'target'] - self.assist_rate * \
                                                                            organization_outputs['target']
        else:
            raise ValueError('Not valid assist')
        return
