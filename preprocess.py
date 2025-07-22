import argparse
import os
import sys
from collections import Counter
from typing import Any, List

import joblib
import pandas as pd
from rdkit import Chem

from utils.generate_edits import generate_reaction_edits


def check_edits(edits: List):
    for edit in edits:  # 遍历编辑列表
        if edit[0] == 'Add Bond':  # 如果编辑类型为'Add Bond'
            return False  # 返回False

    return True  # 如果没有'Add Bond'类型的编辑，则返回True

def preprocessing(rxns: List, args: Any, rxn_classes: List = [], rxns_id=[]) -> None:
    """
    对反应数据进行预处理以获取编辑
    """
    rxns_data = []  # 存储处理后的反应数据
    counter = []  # 计数器
    all_edits = {}  # 存储所有编辑
    rxn_lengths = []

    savedir = f'data/{args.dataset}/{args.mode}'  # 保存目录路径
    os.makedirs(savedir, exist_ok=True)  # 创建保存目录（如果不存在）

    for idx, rxn_smi in enumerate(rxns):  # 遍历反应SMILES列表并获取索引和SMILES字符串
        r, p = rxn_smi.split('>>')  # 将反应SMILES字符串拆分为反应物和生成物
        prod_mol = Chem.MolFromSmiles(p)  # 从生成物SMILES字符串创建分子对象

        if (prod_mol is None) or (prod_mol.GetNumAtoms() <= 1) or (prod_mol.GetNumBonds() <= 1):  # 检查生成物分子是否为空或原子数/键数少于等于1
            print(
                f'Product has 0 or 1 atom or 1 bond, Skipping reaction {idx}')  # 打印跳过具有0或1个原子或1个键的生成物的反应信息
            print()  # 打印空行
            sys.stdout.flush()  # 刷新标准输出缓冲区
            continue  # 继续下一次循环

        react_mol = Chem.MolFromSmiles(r)

        if (react_mol is None) or (react_mol.GetNumAtoms() <= 1) or (prod_mol.GetNumBonds() <= 1):  # 检查反应物分子是否为空或原子数/生成物分子键数少于等于1
            print(
                f'Reactant has 0 or 1 atom or 1 bond, Skipping reaction {idx}')  # 打印跳过具有0或1个原子或1个键的反应物的反应信息
            print()  # 打印空行
            sys.stdout.flush()  # 刷新标准输出缓冲区
            continue  # 继续下一次循环

        try:
            if args.dataset == 'C-H Arylation':  # 如果数据集为'C-H Arylation'
                rxn_data = generate_reaction_edits(rxn_smi, kekulize=args.kekulize, rxn_class=int(
                    rxn_classes[idx]) - 1, rxn_id=rxns_id[idx])  # 生成反应数据
            else:
                rxn_data = generate_reaction_edits(
                    rxn_smi, kekulize=args.kekulize)  # 生成反应数据
        except:
            print(f'Failed to extract reaction data, skipping reaction {idx}')  # 捕获异常，打印提取反应数据失败的信息
            print()  # 打印空行
            sys.stdout.flush()  # 刷新标准输出缓冲区
            continue  # 继续下一次循环

        edits_accepted = check_edits(rxn_data.edits)  # 检查编辑是否被接受
        if not edits_accepted:  # 如果编辑未被接受
            print(f'Edit: Add new bond. Skipping reaction {idx}')  # 打印添加新键的编辑信息，并跳过该反应
            print()  # 打印空行
            sys.stdout.flush()  # 刷新标准输出缓冲区
            continue  # 继续下一次循环

        # if args.dataset == 'uspto_full':
        #     if len(rxn_data.edits) > 9 or len(rxn_data.edits) == 1:
        #         print(f'Edits step exceed max_steps or edit step is 1. Skipping reaction {idx}')
        #         print()
        #         sys.stdout.flush()
        #         continue

        rxns_data.append(rxn_data)  # 将处理后的反应数据添加到列表中
        # 记录反应ID和编辑长度
        rxn_lengths.append({
            'id': rxns_id[idx],  # 原始文件的ID
            'edit_length': len(rxn_data.edits)  # 编辑长度
        })
        lengths_df = pd.DataFrame(rxn_lengths)  # 创建DataFrame
        lengths_file = os.path.join(savedir, f'{args.mode}_edit_lengths.csv')  # 设置文件名
        lengths_df.to_csv(lengths_file, index=False)  # 保存为CSV文件

        # print(f'Edit lengths saved to {lengths_file}')  # 打印保存文件的信息

        if (idx % args.print_every == 0) and idx:  # 每处理 args.print_every 个反应并且不是第一个反应时
            print(f'{idx}/{len(rxns)} {args.mode} reactions processed.')  # 打印已处理的反应数量和模式
            sys.stdout.flush()  # 刷新标准输出缓冲区

    print(f'All {args.mode} reactions complete.')  # 打印所有反应处理完成的信息
    sys.stdout.flush()  # 刷新标准输出缓冲区

    save_file = os.path.join(savedir, f'{args.mode}.file')  # 构建保存文件路径
    if args.kekulize:  # 如果需要进行Kekulization
        save_file += '.kekulized'  # 在文件名后添加'.kekulized'

    if args.mode == 'test':  # 如果模式为'test'
        for idx, rxn_data in enumerate(rxns_data):  # 遍历处理后的反应数据
            for edit in rxn_data.edits:  # 遍历每个反应的编辑
                if edit not in all_edits:  # 如果编辑不在所有编辑中
                    all_edits[edit] = 1  # 将编辑添加到所有编辑中并设置计数为1
                else:
                    all_edits[edit] += 1  # 否则，增加编辑的计数

        atom_edits = []  # 存储原子编辑
        bond_edits = []  # 存储键编辑
        lg_edits = []  # 存储LG编辑
        atom_lg_edits = []  # 存储原子和LG编辑

        if args.dataset == 'C-H Arylation':  # 如果数据集为'C-H Arylation'
            for edit, num in all_edits.items():  # 遍历所有编辑及其计数
                if edit[0] == 'Change Atom':  # 如果编辑类型为'Change Atom'
                    atom_edits.append(edit)  # 将编辑添加到原子编辑列表
                    atom_lg_edits.append(edit)  # 将编辑添加到原子和LG编辑列表
                elif edit[0] == 'Delete Bond' or edit[0] == 'Change Bond' or edit[0] == 'Add Bond':  # 如果编辑类型为'Delete Bond'、'Change Bond'或'Add Bond'
                    bond_edits.append(edit)  # 将编辑添加到键编辑列表
                elif edit[0] == 'Attaching LG':  # 如果编辑类型为'Attaching LG'
                    lg_edits.append(edit)  # 将编辑添加到LG编辑列表
            atom_lg_edits.extend(lg_edits)  # 将LG编辑添加到原子和LG编辑列表

        elif args.dataset == 'uspto_full':  # 如果数据集为'uspto_full'
            for edit, num in all_edits.items():  # 遍历所有编辑及其计数
                if edit[0] == 'Change Atom':  # 如果编辑类型为'Change Atom'
                    atom_edits.append(edit)  # 将编辑添加到原子编辑列表
                    atom_lg_edits.append(edit)  # 将编辑添加到原子和LG编辑列表
                elif edit[0] == 'Delete Bond' or edit[0] == 'Change Bond' or edit[0] == 'Add Bond':  # 如果编辑类型为'Delete Bond'、'Change Bond'或'Add Bond'
                    bond_edits.append(edit)  # 将编辑添加到键编辑列表
                elif edit[0] == 'Attaching LG' and num >= 50:  # 如果编辑类型为'Attaching LG'且计数大于等于50
                    lg_edits.append(edit)  # 将编辑添加到LG编辑列表
            atom_lg_edits.extend(lg_edits)  # 将LG编辑添加到原子和LG编辑列表

        print(atom_edits)  # 打印原子编辑列表
        print(bond_edits)  # 打印键编辑列表
        print(lg_edits)  # 打印LG编辑列表

        filter_rxns_data = []  # 存储筛选后的反应数据
        for idx, rxn_data in enumerate(rxns_data):  # 遍历处理后的反应数据
            for edit in rxn_data.edits:  # 遍历每个反应的编辑
                if edit[0] == 'Attaching LG' and edit not in lg_edits:  # 如果编辑类型为'Attaching LG'且不在LG编辑列表中
                    print(
                        f'The number of {edit} in training set is very small, skipping reaction')  # 打印在训练集中{edit}的数量非常少，跳过该反应
                    rxn_data = None  # 将反应数据设为None
            if rxn_data is not None:  # 如果反应数据不为None
                counter.append(len(rxn_data.edits))  # 将反应数据的编辑数量添加到计数器中
                filter_rxns_data.append(rxn_data)  # 将反应数据添加到筛选后的反应数据中

        print(Counter(counter))  # 打印计数器中的统计信息

        joblib.dump(filter_rxns_data, save_file, compress=3)  # 将筛选后的反应数据保存到文件中
        joblib.dump(atom_edits, os.path.join(savedir, 'atom_vocab.txt'))  # 将原子编辑保存到文件中
        joblib.dump(bond_edits, os.path.join(savedir, 'bond_vocab.txt'))  # 将键编辑保存到文件中
        joblib.dump(lg_edits, os.path.join(savedir, 'lg_vocab.txt'))  # 将LG编辑保存到文件中
        joblib.dump(atom_lg_edits, os.path.join(savedir, 'atom_lg_vocab.txt'))  # 将原子和LG编辑保存到文件中
    else:
        bond_vocab_file = f'data/{args.dataset}/train/bond_vocab.txt'  # 设置键词汇文件路径
        atom_vocab_file = f'data/{args.dataset}/train/atom_lg_vocab.txt'  # 设置原子和LG词汇文件路径
        bond_vocab = joblib.load(bond_vocab_file)  # 加载键词汇文件
        atom_vocab = joblib.load(atom_vocab_file)  # 加载原子和LG词汇文件
        bond_vocab.extend(atom_vocab)  # 将原子和键词汇合并
        all_edits = bond_vocab  # 更新所有编辑为合并后的词汇

        cover_num = 0  # 初始化覆盖数量
        for idx, rxn_data in enumerate(rxns_data):  # 遍历处理后的反应数据
            cover = True  # 初始化覆盖标志为True
            for edit in rxn_data.edits:  # 遍历每个反应的编辑
                if edit != 'Terminate' and edit not in all_edits:  # 如果编辑不是'Terminate'且不在所有编辑中
                    print(f'{edit} in {args.mode} is not in train set')  # 打印不在训练集中的编辑信息
                    cover = False  # 将覆盖标志设为False
            if cover:  # 如果覆盖标志为True
                cover_num += 1  # 增加覆盖数量

            counter.append(len(rxn_data.edits))  # 将反应数据的编辑数量添加到计数器中

        print(Counter(counter))  # 打印计数器中的统计信息
        print(f'The cover rate is {cover_num}/{len(rxns_data)}')  # 打印覆盖率信息
        joblib.dump(rxns_data, save_file, compress=3)  # 将处理后的反应数据保存到文件中


def main():
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument('--dataset', type=str, default='C-H Arylation',
                        help='dataset: C-H Arylation or uspto_full')  # 添加数据集参数
    parser.add_argument('--mode', type=str, default='test',
                        help='Type of dataset being prepared: train or valid or test')  # 添加数据集类型参数
    parser.add_argument('--print_every', type=int,
                        default=1000, help='Print during preprocessing')  # 添加打印频率参数
    parser.add_argument('--kekulize', default=True, action='store_true',
                        help='Whether to kekulize mols during training')  # 添加Kekulize参数
    args = parser.parse_args()  # 解析命令行参数

    args.dataset = args.dataset.lower()  # 将数据集名称转换为小写
    datadir = f'data/{args.dataset}/'  # 设置数据目录路径
    rxn_key = 'mapped_rxn'  # 设置反应键
    if args.dataset == 'C-H Arylation':  # 如果数据集为'C-H Arylation'
        filename = f'canonicalized_{args.mode}.csv'  # 构建文件名
        df = pd.read_csv(os.path.join(datadir, filename))  # 读取CSV文件
        preprocessing(rxns=df[rxn_key], args=args, rxn_classes=df['class'], rxns_id=df['id'])  # 进行预处理
    else:
        filename = f'raw_{args.mode}.csv'  # 构建文件名
        df = pd.read_csv(os.path.join(datadir, filename))  # 读取CSV文件
        preprocessing(rxns=df[rxn_key], args=args)  # 进行预处理


if __name__ == '__main__':
    main()

