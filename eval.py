import numpy as np
import pandas as pd
import os
import argparse
import joblib
from tqdm import tqdm
from collections import Counter
import torch
from rdkit import Chem, RDLogger

from models import TransGraphEdit, BeamSearch
lg = RDLogger.logger()
lg.setLevel(4)

ROOT_DIR = './'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def canonicalize(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
    except:
        print('no mol', flush=True)
        return smi
    if mol is None:
        return smi
    mol = Chem.RemoveHs(mol)
    [a.ClearProp('molAtomMapNumber') for a in mol.GetAtoms()]
    return Chem.MolToSmiles(mol)


def canonicalize_p(smi):
    p = canonicalize(smi)
    p_mol = Chem.MolFromSmiles(p)
    [a.SetAtomMapNum(a.GetIdx()+1) for a in p_mol.GetAtoms()]
    p_smi = Chem.MolToSmiles(p_mol)
    return p_smi

# ROOT_DIR = 'data/C-H Arylation/canonicalized_test-1.csv'
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='C-H Arylation',
                        help='dataset: C-H Arylation or USPTO_full')
    parser.add_argument("--use_rxn_class", default=False,
                        action='store_true', help='Whether to use rxn_class')
    parser.add_argument('--experiments', type=str, default='15-11-2024--10-36-38',
                        help='Name of edits prediction experiment')
    parser.add_argument('--beam_size', type=int,
                        default=10, help='Beam search width')
    parser.add_argument('--max_steps', type=int, default=9,
                        help='maximum number of edit steps')

    args = parser.parse_args()
    args.dataset = args.dataset.lower()

    data_dir = os.path.join(ROOT_DIR, 'data', f'{args.dataset}', 'test')
    test_file = os.path.join(data_dir, 'test.file.kekulized')
    test_data = joblib.load(test_file)
    if args.use_rxn_class:
        exp_dir = os.path.join(
            ROOT_DIR, 'experiments', f'{args.dataset}', 'with_rxn_class', f'{args.experiments}')
    else:
        exp_dir = os.path.join(
            ROOT_DIR, 'experiments', f'{args.dataset}', 'without_rxn_class', f'{args.experiments}')

    checkpoint = torch.load(os.path.join(exp_dir, 'epoch_78.pt'))
    config = checkpoint['saveables']

    model = TransGraphEdit(**config, device=DEVICE)
    model.load_state_dict(checkpoint['state'])
    model.to(DEVICE)
    model.eval()

    top_k = np.zeros(args.beam_size)
    edit_steps_cor = []
    counter = []
    stereo_rxn = []
    stereo_rxn_cor = []
    beam_model = BeamSearch(model=model, step_beam_size=10,
                            beam_size=args.beam_size, use_rxn_class=args.use_rxn_class)
    p_bar = tqdm(list(range(len(test_data))))
    print("Number of reactions in test_data:", len(test_data))

    pred_file = os.path.join(exp_dir, 'pred_results.txt')
    file_num = 1
    while os.path.exists(pred_file):
        pred_file = os.path.join(exp_dir, f'pred_results_{file_num}.txt')
        file_num += 1

    average_scores = []  # List to store the average scores
    incorrect_top1 = []  # List to store incorrect top-1 predictions
    incorrect_top10 = []  # List to store incorrect top-10 predictions
    individual_top_k_scores = []

    with open(pred_file, 'a') as fp:
        for idx in p_bar:
            rxn_data = test_data[idx]
            rxn_smi = rxn_data.rxn_smi
            rxn_class = rxn_data.rxn_class
            edit_steps = len(rxn_data.edits)  # 计算rxn_data中编辑步骤的数量，并将其赋值给edit_steps
            counter.append(edit_steps)  # 将编辑步骤的数量添加到counter列表中，用于后续统计或分析

            r, p = rxn_smi.split('>>')
            r_mol = Chem.MolFromSmiles(r)
            [a.ClearProp('molAtomMapNumber') for a in r_mol.GetAtoms()]
            r_mol = Chem.MolFromSmiles(Chem.MolToSmiles(r_mol))
            r_smi = Chem.MolToSmiles(r_mol, isomericSmiles=True)
            r_set = set(r_smi.split('.'))   # 将r_smi字符串按'.'分割成多个部分

            with torch.no_grad():
                top_k_results = beam_model.run_search(
                    prod_smi=p, max_steps=args.max_steps, rxn_class=rxn_class)

            fp.write(f'({idx}) {rxn_smi}\n')

            beam_matched = False
            correct_top1 = False  # Initialize correct_top1 as False
            correct_top10 = False  # Initialize correct_top10 as False
            top_k_scores = []

            for beam_idx, path in enumerate(top_k_results):
                pred_smi = path['final_smi']
                prob = path['prob']
                pred_set = set(pred_smi.split('.'))
                correct = pred_set == r_set  # 如果预测的和反应物一样则correct为True
                top_k_scores.append(1 if correct else 0)
                str_edits = '|'.join(f'({str(edit)};{p})'for edit, p in zip(
                    path['rxn_actions'], path['edits_prob']))
                fp.write(
                    f'{beam_idx} prediction_is_correct:{correct} probability:{prob} {pred_smi} {str_edits}\n')
                if correct and not beam_matched:
                    top_k[beam_idx] += 1
                    beam_matched = True
                    if beam_idx == 0:
                        correct_top1 = True
                    if beam_idx < 10:
                        correct_top10 = True
            individual_top_k_scores.append(top_k_scores)

            fp.write('\n')
            if beam_matched:
                edit_steps_cor.append(edit_steps)
            else:
                correct_top1 = False
                correct_top10 = False

            if not correct_top1:
                incorrect_top1.append((rxn_smi, top_k_results[0]['final_smi']))
            if not correct_top10:
                incorrect_top10.append((rxn_smi, [path['final_smi'] for path in top_k_results[:10]]))

            for edit in rxn_data.edits:
                if edit[1] == (1, 1) or edit[1] == (1, 2) or edit[1] == (0, 1) or edit[1] == (0, 2) or edit[1] == (2, 2) or edit[1] == (2, 3):
                    stereo_rxn.append(idx)
                    if beam_matched:
                        stereo_rxn_cor.append(idx)

            msg = 'average score'
            avg_scores = {'id': idx}
            for beam_idx in [1, 3, 5, 10, 50]:
                match_acc = np.sum(top_k[:beam_idx]) / (idx + 1)
                msg += ', t%d: %.4f' % (beam_idx, match_acc)
                avg_scores[f't{beam_idx}'] = match_acc
            p_bar.set_description(msg)
            average_scores.append(avg_scores)  # Append the average scores for this iteration

        edit_steps = Counter(counter)
        edit_steps_correct = Counter(edit_steps_cor)
        fp.write(f'edit_steps_reaction_number:{edit_steps}\n')
        fp.write(f'edit_steps_reaction_prediction_correct:{edit_steps_correct}\n')
        fp.write(f'stereo_reaction_idx:{stereo_rxn}\n')
        fp.write((f'stereo_reaction_prediction_correct:{stereo_rxn_cor}\n'))

    # Save average scores to CSV
    avg_scores_df = pd.DataFrame(average_scores)
    avg_scores_df.to_csv(os.path.join(exp_dir, 'average_scores.csv'), index=False)
    # 保存每个反应的单独得分到 CSV
    individual_top_k_df = pd.DataFrame(individual_top_k_scores, columns=[f'Top-{i + 1}' for i in range(args.beam_size)])
    individual_top_k_df.to_csv(os.path.join(exp_dir, 'individual_top_k_scores.csv'), index=False)

    # Save incorrect predictions to CSV
    incorrect_top1_df = pd.DataFrame(incorrect_top1, columns=['Reaction', 'Top-1 Prediction'])
    incorrect_top1_df.to_csv(os.path.join(exp_dir, 'incorrect_top1_predictions.csv'), index=False)

    incorrect_top10_df = pd.DataFrame(incorrect_top10, columns=['Reaction', 'Top-10 Predictions'])
    incorrect_top10_df.to_csv(os.path.join(exp_dir, 'incorrect_top10_predictions.csv'), index=False)


if __name__ == '__main__':
    main()


