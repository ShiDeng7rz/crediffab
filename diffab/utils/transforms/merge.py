import torch

from ._base import register_transform
from ..protein import constants


def assign_chain_number_(data_list):
    chains = set()
    for data in data_list:
        chains.update(data['chain_id'])
    chains = {c: i for i, c in enumerate(chains)}

    for data in data_list:
        data['chain_nb'] = torch.LongTensor([
            chains[c] for c in data['chain_id']
        ])


def _data_attr(data, name):
    if name in ('generate_flag', 'anchor_flag') and name not in data:
        return torch.zeros(data['aa'].shape, dtype=torch.bool)
    else:
        return data[name]


@register_transform('merge_chains')
class MergeChains(object):

    def __init__(self):
        super().__init__()

    def __call__(self, structure):
        data_list = []
        if structure['heavy'] is not None:
            structure['heavy']['fragment_type'] = torch.full_like(
                structure['heavy']['aa'],
                fill_value=constants.Fragment.Heavy,
            )
            data_list.append(structure['heavy'])

        if structure['light'] is not None:
            structure['light']['fragment_type'] = torch.full_like(
                structure['light']['aa'],
                fill_value=constants.Fragment.Light,
            )
            data_list.append(structure['light'])

        if structure['antigen'] is not None:
            structure['antigen']['fragment_type'] = torch.full_like(
                structure['antigen']['aa'],
                fill_value=constants.Fragment.Antigen,
            )
            structure['antigen']['cdr_flag'] = torch.zeros_like(
                structure['antigen']['aa'],
            )
            data_list.append(structure['antigen'])

        assign_chain_number_(data_list)

        list_props = {
            'chain_id': [],
            'icode': [],
        }
        tensor_props = {
            'chain_nb': [],
            'resseq': [],
            'res_nb': [],
            'aa': [],
            'pos_heavyatom': [],
            'mask_heavyatom': [],
            'generate_flag': [],
            'cdr_flag': [],
            'anchor_flag': [],
            'fragment_type': [],
        }

        for data in data_list:
            for k in list_props.keys():
                list_props[k].append(_data_attr(data, k))
            for k in tensor_props.keys():
                tensor_props[k].append(_data_attr(data, k))

        list_props = {k: sum(v, start=[]) for k, v in list_props.items()}
        tensor_props = {k: torch.cat(v, dim=0) for k, v in tensor_props.items()}
        data_out = {
            **list_props,
            **tensor_props,
        }
        return data_out


@register_transform('split_chains')
class Split_Chains(object):

    def __init__(self):
        super().__init__()

    def __call__(self, structure):
        # 先给现有三部分打上 fragment_type / cdr_flag（与原逻辑一致）
        data_heavy = structure['heavy']
        data_light = structure['light']
        data_ag = structure['antigen']
        complex = []
        if data_heavy is not None:
            data_heavy['fragment_type'] = torch.full_like(
                data_heavy['aa'], fill_value=constants.Fragment.Heavy
            )
            complex.append(data_heavy)
        if data_light is not None:
            data_light['fragment_type'] = torch.full_like(
                data_light['aa'], fill_value=constants.Fragment.Light
            )
            complex.append(data_light)
        if data_ag is not None:
            data_ag['fragment_type'] = torch.full_like(
                data_ag['aa'], fill_value=constants.Fragment.Antigen
            )
            # 抗原没有 CDR，置 0
            data_ag['cdr_flag'] = torch.zeros_like(data_ag['aa'])
            complex.append(data_ag)

        assign_chain_number_(complex)
        list_props = {
            'chain_id': [],
            'icode': [],
        }
        tensor_props = {
            'chain_nb': [],
            'resseq': [],
            'res_nb': [],
            'aa': [],
            'pos_heavyatom': [],
            'mask_heavyatom': [],
            'generate_flag': [],
            'cdr_flag': [],
            'anchor_flag': [],
            'fragment_type': [],
        }

        for data in complex:
            for k in list_props.keys():
                list_props[k].append(_data_attr(data, k))
            for k in tensor_props.keys():
                tensor_props[k].append(_data_attr(data, k))

        list_props = {k: sum(v, start=[]) for k, v in list_props.items()}
        tensor_props = {k: torch.cat(v, dim=0) for k, v in tensor_props.items()}
        data_out = {
            **list_props,
            **tensor_props,
        }

        # ===== 选项 A：分别为“抗体组(heavy+light)”与“抗原组(antigen)”分配 chain_nb（各自从 0 编号）=====
        ab_list = [x for x in (data_heavy, data_light) if x is not None]
        ag_list = [x for x in (data_ag,) if x is not None]

        if ab_list:
            assign_chain_number_(ab_list)  # 仅在抗体内部编号（如 H=0, L=1）
        if ag_list:
            assign_chain_number_(ag_list)  # 抗原内部编号（仅一条链时通常全 0）

        # 如果你希望“全局一致编号”（例如 H=0, L=1, Antigen=2），改为把三者合在一起调用一次：
        # all_list = [x for x in (data_heavy, data_light, data_ag) if x is not None]
        # if all_list:
        #     self.assign_chain_number_(all_list)

        # 组装函数：把若干链打平成一份 data_out（与你原来的逻辑完全一致）
        def _assemble(list_of_parts):
            if not list_of_parts:
                return None

            list_props = {'chain_id': [], 'icode': []}
            tensor_props = {
                'chain_nb': [], 'resseq': [], 'res_nb': [], 'aa': [],
                'pos_heavyatom': [], 'mask_heavyatom': [],
                'generate_flag': [], 'cdr_flag': [], 'anchor_flag': [],
                'fragment_type': [], 'A': []
            }

            for data in list_of_parts:
                for k in list_props.keys():
                    list_props[k].append(_data_attr(data, k))
                for k in tensor_props.keys():
                    tensor_props[k].append(_data_attr(data, k))

            list_props = {k: sum(v, start=[]) for k, v in list_props.items()}
            tensor_props = {k: torch.cat(v, dim=0) for k, v in tensor_props.items()}
            return {**list_props, **tensor_props}

        # 分别组装
        ab_out = _assemble(ab_list)  # heavy+light 合并后的样本；若两者皆无则为 None
        ag_out = _assemble(ag_list)  # antigen 的样本；若无 antigen 则为 None

        # 你可以选择返回二元组或字典；这里用字典更直观
        return {'antibody': ab_out, 'antigen': ag_out, 'complex': data_out}
