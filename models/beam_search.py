import numpy as np
import numpy as np
from typing import List
import torch
import torch.nn.functional as F
from rdkit import Chem

from utils.rxn_graphs import MolGraph
from utils.collate_fn import get_batch_graphs
from prepare_data import apply_edit_to_mol
from utils.reaction_actions import (AddGroupAction, AtomEditAction,
                                    BondEditAction, Termination)


class BeamSearch:
    def __init__(self, model, step_beam_size, beam_size, use_rxn_class):
        self.model = model
        self.step_beam_size = step_beam_size
        self.beam_size = beam_size
        self.use_rxn_class = use_rxn_class

    def process_path(self, path, rxn_class):
        new_paths = []

        prod_mol = path['prod_mol']
        steps = path['steps'] + 1
        prod_tensors = self.model.to_device(path['tensors'])
        edit_logits, state, state_scope = self.model.compute_edit_scores(
            prod_tensors, path['scopes'], path['state'], path['state_scope'])
        edit_logits = edit_logits[0]
        edit_logits = F.softmax(edit_logits, dim=-1)

        k = self.step_beam_size
        top_k_vals, top_k_idxs = torch.topk(edit_logits, k=k)

        for beam_idx, (topk_idx, val) in enumerate(zip(*(top_k_idxs, top_k_vals))):
            edit, edit_atom = self.get_edit_from_logits(
                mol=prod_mol, edit_logits=edit_logits, idx=topk_idx, val=val)
            val = round(val.item(), 4)
            new_prob = path['prob'] * val

            if edit == 'Terminate':
                edits_prob, edits = [], []
                edits_prob.extend(path['edits_prob'])
                edits_prob.append(val)
                edits.extend(path['edits'])
                edits.append(edit)
                final_path = {
                    'prod_mol': prod_mol,
                    'steps': steps,
                    'prob': new_prob,
                    'edits_prob': edits_prob,
                    'tensors': path['tensors'],
                    'scopes': path['scopes'],
                    'state': state,
                    'state_scope': state_scope,
                    'edits': edits,
                    'edits_atom': path['edits_atom'],
                    'finished': True,
                }
                new_paths.append(final_path)

            else:
                try:
                    int_mol = apply_edit_to_mol(mol=Chem.Mol(
                        prod_mol), edit=edit, edit_atom=edit_atom)
                    prod_graph = MolGraph(mol=Chem.Mol(
                        int_mol), rxn_class=rxn_class, use_rxn_class=self.use_rxn_class)
                    prod_tensors, prod_scopes = get_batch_graphs(
                        [prod_graph], use_rxn_class=self.use_rxn_class)
                    edits_prob, edits, edits_atom = [], [], []
                    edits_prob.extend(path['edits_prob'])
                    edits_prob.append(val)
                    edits.extend(path['edits'])
                    edits.append(edit)
                    edits_atom.extend(path['edits_atom'])
                    edits_atom.append(edit_atom)
                    new_path = {
                        'prod_mol': int_mol,
                        'steps': steps,
                        'prob': new_prob,
                        'edits_prob': edits_prob,
                        'tensors': prod_tensors,
                        'scopes': prod_scopes,
                        'state': state,
                        'state_scope': state_scope,
                        'edits': edits,
                        'edits_atom': edits_atom,
                        'finished': False,
                    }
                    new_paths.append(new_path)
                except:
                    continue

        return new_paths

    def get_top_k_paths(self, paths):
        k = min(len(paths), self.beam_size)
        path_argsort = np.argsort([-path['prob'] for path in paths])
        filtered_paths = [paths[i] for i in path_argsort[:k]]

        return filtered_paths

    def get_edit_from_logits(self, mol, edit_logits, idx, val):
        max_bond_idx = mol.GetNumBonds() * self.model.bond_outdim

        if idx.item() == len(edit_logits) - 1:
            edit = 'Terminate'
            edit_atom = []

        elif idx.item() < max_bond_idx:
            bond_logits = edit_logits[:mol.GetNumBonds(
            ) * self.model.bond_outdim]
            bond_logits = bond_logits.reshape(
                mol.GetNumBonds(), self.model.bond_outdim)
            idx_tensor = torch.where(bond_logits == val)

            idx_tensor = [indices[-1] for indices in idx_tensor]

            bond_idx, edit_idx = idx_tensor[0].item(), idx_tensor[1].item()
            a1 = mol.GetBondWithIdx(bond_idx).GetBeginAtom().GetAtomMapNum()
            a2 = mol.GetBondWithIdx(bond_idx).GetEndAtom().GetAtomMapNum()

            a1, a2 = sorted([a1, a2])
            edit_atom = [a1, a2]
            edit = self.model.bond_vocab.get_elem(edit_idx)

        else:
            atom_logits = edit_logits[max_bond_idx:-1]

            assert len(atom_logits) == mol.GetNumAtoms() * \
                self.model.atom_outdim
            atom_logits = atom_logits.reshape(
                mol.GetNumAtoms(), self.model.atom_outdim)
            idx_tensor = torch.where(atom_logits == val)

            idx_tensor = [indices[-1] for indices in idx_tensor]
            atom_idx, edit_idx = idx_tensor[0].item(), idx_tensor[1].item()

            a1 = mol.GetAtomWithIdx(atom_idx).GetAtomMapNum()
            edit_atom = a1
            edit = self.model.atom_vocab.get_elem(edit_idx)

        return edit, edit_atom

    def run_search(self, prod_smi: str, max_steps: int = 8, rxn_class: int = None) -> List[dict]:
        product = Chem.MolFromSmiles(prod_smi)
        Chem.Kekulize(product)
        prod_graph = MolGraph(mol=Chem.Mol(
            product), rxn_class=rxn_class, use_rxn_class=self.use_rxn_class)
        prod_tensors, prod_scopes = get_batch_graphs(
            [prod_graph], use_rxn_class=self.use_rxn_class)

        paths = []
        start_path = {
            'prod_mol': product,
            'steps': 0,
            'prob': 1.0,
            'edits_prob': [],
            'tensors': prod_tensors,
            'scopes': prod_scopes,
            'state': None,
            'state_scope': None,
            'edits': [],
            'edits_atom': [],
            'finished': False,
        }
        paths.append(start_path)

        for step_i in range(max_steps):
            followed_path = [path for path in paths if not path['finished']]
            if len(followed_path) == 0:
                break

            paths = [path for path in paths if path['finished']]

            for path in followed_path:
                new_paths = self.process_path(path, rxn_class)
                paths += new_paths

            paths = self.get_top_k_paths(paths)

            if all(path['finished'] for path in paths):
                break

        finished_paths = []
        for path in paths:
            if path['finished']:
                try:
                    int_mol = product
                    path['rxn_actions'] = []
                    for i, edit in enumerate(path['edits']):
                        if int_mol is None:
                            print("Interim mol is None")
                            break
                        if edit == 'Terminate':
                            edit_exe = Termination(action_vocab='Terminate')
                            path['rxn_actions'].append(edit_exe)
                            pred_mol = edit_exe.apply(int_mol)
                            [a.ClearProp('molAtomMapNumber')
                             for a in pred_mol.GetAtoms()]
                            pred_mol = Chem.MolFromSmiles(
                                Chem.MolToSmiles(pred_mol))
                            final_smi = Chem.MolToSmiles(pred_mol)
                            path['final_smi'] = final_smi

                        elif edit[0] == 'Change Atom':
                            edit_exe = AtomEditAction(
                                path['edits_atom'][i], *edit[1], action_vocab='Change Atom')
                            path['rxn_actions'].append(edit_exe)
                            int_mol = edit_exe.apply(int_mol)

                        elif edit[0] == 'Delete Bond':
                            edit_exe = BondEditAction(
                                *path['edits_atom'][i], *edit[1], action_vocab='Delete Bond')
                            path['rxn_actions'].append(edit_exe)
                            int_mol = edit_exe.apply(int_mol)

                        if edit[0] == 'Change Bond':
                            edit_exe = BondEditAction(
                                *path['edits_atom'][i], *edit[1], action_vocab='Change Bond')
                            path['rxn_actions'].append(edit_exe)
                            int_mol = edit_exe.apply(int_mol)

                        if edit[0] == 'Attaching LG':
                            edit_exe = AddGroupAction(
                                path['edits_atom'][i], edit[1], action_vocab='Attaching LG')
                            path['rxn_actions'].append(edit_exe)
                            int_mol = edit_exe.apply(int_mol)

                    finished_paths.append(path)

                except Exception as e:
                    print(f'Exception while final mol to Smiles: {str(e)}')
                    path['final_smi'] = 'final_smi_unmapped'
                    finished_paths.append(path)

        return finished_paths

# import numpy as np
# from typing import List
# import torch
# import torch.nn.functional as F
# from rdkit import Chem
#
# from utils.rxn_graphs import MolGraph
# from utils.collate_fn import get_batch_graphs
# from prepare_data import apply_edit_to_mol
# from utils.reaction_actions import (AddGroupAction, AtomEditAction,
#                                     BondEditAction, Termination)
#
#
# # {'Delete Bond': 0.32202548403111037, 'Change Bond': 0.12344861823597551, 'Attaching LG': 0.2528545424458051, 'Terminate': 0.2701059076617574, 'Change Atom': 0.03156544762535165}
# edit_prior_probabilities = {
#     'Delete Bond': 0.32202548403111037,
#     'Change Bond': 0.12344861823597551,
#     'Attaching LG': 0.2528545424458051,
#     'Terminate': 0.2701059076617574,
#     'Change Atom': 0.03156544762535165,
# }
# class BeamSearch:
#     def __init__(self, model, step_beam_size, beam_size, use_rxn_class):
#         self.model = model
#         self.step_beam_size = step_beam_size
#         self.beam_size = beam_size
#         self.use_rxn_class = use_rxn_class
#         self.prior_probs = {
#             'Delete Bond': 0.32202548403111037,
#             'Change Bond': 0.12344861823597551,
#             'Attaching LG': 0.2528545424458051,
#             'Terminate': 0.2701059076617574,
#             'Change Atom': 0.03156544762535165
#         }
#
#     def process_path(self, path, rxn_class):
#         new_paths = []
#
#         prod_mol = path['prod_mol']
#         steps = path['steps'] + 1
#         prod_tensors = self.model.to_device(path['tensors'])
#         edit_logits, state, state_scope = self.model.compute_edit_scores(
#             prod_tensors, path['scopes'], path['state'], path['state_scope'])
#         edit_logits = edit_logits[0]
#         edit_logits = F.softmax(edit_logits, dim=-1)
#
#         k = self.step_beam_size
#         top_k_vals, top_k_idxs = torch.topk(edit_logits, k=k)
#
#         # 用于标记是否已经执行了删除键操作
#         delete_bond_done = False
#
#         for beam_idx, (topk_idx, val) in enumerate(zip(*(top_k_idxs, top_k_vals))):
#             edit, edit_atom = self.get_edit_from_logits(
#                 mol=prod_mol, edit_logits=edit_logits, idx=topk_idx, val=val)
#             val = round(val.item(), 4)
#             new_prob = path['prob'] * val
#
#             # 如果还未执行删除键操作且选中的操作是删除键
#             if not delete_bond_done and edit == 'Delete Bond':
#                 delete_bond_done = True  # 标记为已执行删除键操作
#
#                 # 创建删除键的新路径
#                 new_path = {
#                     'prod_mol': prod_mol,
#                     'steps': steps,
#                     'prob': new_prob,
#                     'edits_prob': [val],  # 只保存删除键的概率
#                     'tensors': path['tensors'],
#                     'scopes': path['scopes'],
#                     'state': state,
#                     'state_scope': state_scope,
#                     'edits': [edit],
#                     'edits_atom': [edit_atom],
#                     'finished': False,
#                 }
#                 new_paths.append(new_path)
#
#             # 如果已经执行了删除键操作，则查找其他操作
#             if delete_bond_done:
#                 # 如果编辑是终止操作
#                 if edit == 'Terminate':
#                     final_path = {
#                         'prod_mol': prod_mol,
#                         'steps': steps,
#                         'prob': new_prob,
#                         'edits_prob': path['edits_prob'] + [val],  # 保存所有编辑的概率
#                         'tensors': path['tensors'],
#                         'scopes': path['scopes'],
#                         'state': state,
#                         'state_scope': state_scope,
#                         'edits': path['edits'] + [edit],  # 添加Terminate操作
#                         'edits_atom': path['edits_atom'] + [edit_atom],
#                         'finished': True,
#                     }
#                     new_paths.append(final_path)
#
#                 # 否则添加其他编辑操作
#                 else:
#                     new_path = {
#                         'prod_mol': prod_mol,
#                         'steps': steps,
#                         'prob': new_prob,
#                         'edits_prob': path['edits_prob'] + [val],  # 保存所有编辑的概率
#                         'tensors': path['tensors'],
#                         'scopes': path['scopes'],
#                         'state': state,
#                         'state_scope': state_scope,
#                         'edits': path['edits'] + [edit],  # 添加当前操作
#                         'edits_atom': path['edits_atom'] + [edit_atom],
#                         'finished': False,
#                     }
#                     new_paths.append(new_path)
#
#         return new_paths
#
#     # def adjust_probs(self, model_probs, mol):
#     #     """Adjust the model's probabilities using prior information."""
#     #     adjusted_probs = np.copy(model_probs)
#     #
#     #     # Define the maximum index for bond edits
#     #     max_bond_idx = mol.GetNumBonds() * self.model.bond_outdim
#     #
#     #     # Adjust probabilities for each edit type
#     #     for i in range(len(model_probs)):
#     #         if i < max_bond_idx:  # Assuming this range corresponds to 'Delete Bond' and 'Change Bond'
#     #             if i % 2 == 0:  # Example condition for 'Change Bond' (based on your logic)
#     #                 adjusted_probs[i] *= self.prior_probs.get('Change Bond', 1.0)
#     #             else:  # Assuming odd indices correspond to 'Delete Bond'
#     #                 adjusted_probs[i] *= self.prior_probs.get('Delete Bond', 1.0)
#     #         elif i == max_bond_idx:  # Assuming this corresponds to 'Attaching LG'
#     #             adjusted_probs[i] *= self.prior_probs.get('Attaching LG', 1.0)
#     #         elif i == max_bond_idx + 1:  # Assuming this corresponds to 'Terminate'
#     #             adjusted_probs[i] *= self.prior_probs.get('Terminate', 1.0)
#     #         elif i == max_bond_idx + 2:  # Assuming this corresponds to 'Change Atom'
#     #             adjusted_probs[i] *= self.prior_probs.get('Change Atom', 1.0)
#     #
#     #     # Normalize adjusted probabilities to sum to 1
#     #     adjusted_probs /= np.sum(adjusted_probs)
#     #
#     #     return adjusted_probs
#
#     def get_top_k_paths(self, paths):
#         k = min(len(paths), self.beam_size)
#         path_argsort = np.argsort([-path['prob'] for path in paths])
#         filtered_paths = [paths[i] for i in path_argsort[:k]]
#
#         return filtered_paths
#
#     def get_edit_from_logits(self, mol, edit_logits, idx, val):
#         max_bond_idx = mol.GetNumBonds() * self.model.bond_outdim
#
#         if idx.item() == len(edit_logits) - 1:
#             edit = 'Terminate'
#             edit_atom = []
#
#         elif idx.item() < max_bond_idx:
#             bond_logits = edit_logits[:mol.GetNumBonds(
#             ) * self.model.bond_outdim]
#             bond_logits = bond_logits.reshape(
#                 mol.GetNumBonds(), self.model.bond_outdim)
#             idx_tensor = torch.where(bond_logits == val)
#
#             idx_tensor = [indices[-1] for indices in idx_tensor]
#
#             bond_idx, edit_idx = idx_tensor[0].item(), idx_tensor[1].item()
#             a1 = mol.GetBondWithIdx(bond_idx).GetBeginAtom().GetAtomMapNum()
#             a2 = mol.GetBondWithIdx(bond_idx).GetEndAtom().GetAtomMapNum()
#
#             a1, a2 = sorted([a1, a2])
#             edit_atom = [a1, a2]
#             edit = self.model.bond_vocab.get_elem(edit_idx)
#
#         else:
#             atom_logits = edit_logits[max_bond_idx:-1]
#
#             assert len(atom_logits) == mol.GetNumAtoms() * \
#                 self.model.atom_outdim
#             atom_logits = atom_logits.reshape(
#                 mol.GetNumAtoms(), self.model.atom_outdim)
#             idx_tensor = torch.where(atom_logits == val)
#
#             idx_tensor = [indices[-1] for indices in idx_tensor]
#             atom_idx, edit_idx = idx_tensor[0].item(), idx_tensor[1].item()
#
#             a1 = mol.GetAtomWithIdx(atom_idx).GetAtomMapNum()
#             edit_atom = a1
#             edit = self.model.atom_vocab.get_elem(edit_idx)
#
#         return edit, edit_atom
#
#     def run_search(self, prod_smi: str, max_steps: int = 8, rxn_class: int = None) -> List[dict]:
#         product = Chem.MolFromSmiles(prod_smi)
#         Chem.Kekulize(product)
#         prod_graph = MolGraph(mol=Chem.Mol(
#             product), rxn_class=rxn_class, use_rxn_class=self.use_rxn_class)
#         prod_tensors, prod_scopes = get_batch_graphs(
#             [prod_graph], use_rxn_class=self.use_rxn_class)
#
#         paths = []
#         start_path = {
#             'prod_mol': product,
#             'steps': 0,
#             'prob': 1.0,
#             'edits_prob': [],
#             'tensors': prod_tensors,
#             'scopes': prod_scopes,
#             'state': None,
#             'state_scope': None,
#             'edits': [],
#             'edits_atom': [],
#             'finished': False,
#         }
#         paths.append(start_path)
#
#         for step_i in range(max_steps):
#             followed_path = [path for path in paths if not path['finished']]
#             if len(followed_path) == 0:
#                 break
#
#             paths = [path for path in paths if path['finished']]
#
#             for path in followed_path:
#                 new_paths = self.process_path(path, rxn_class)
#                 paths += new_paths
#
#             paths = self.get_top_k_paths(paths)
#
#             if all(path['finished'] for path in paths):
#                 break
#
#         finished_paths = []
#         for path in paths:
#             if path['finished']:
#                 try:
#                     int_mol = product
#                     path['rxn_actions'] = []
#                     for i, edit in enumerate(path['edits']):
#                         if int_mol is None:
#                             print("Interim mol is None")
#                             break
#                         if edit == 'Terminate':
#                             edit_exe = Termination(action_vocab='Terminate')
#                             path['rxn_actions'].append(edit_exe)
#                             pred_mol = edit_exe.apply(int_mol)
#                             [a.ClearProp('molAtomMapNumber')
#                              for a in pred_mol.GetAtoms()]
#                             pred_mol = Chem.MolFromSmiles(
#                                 Chem.MolToSmiles(pred_mol))
#                             final_smi = Chem.MolToSmiles(pred_mol)
#                             path['final_smi'] = final_smi
#
#                         elif edit[0] == 'Change Atom':
#                             edit_exe = AtomEditAction(
#                                 path['edits_atom'][i], *edit[1], action_vocab='Change Atom')
#                             path['rxn_actions'].append(edit_exe)
#                             int_mol = edit_exe.apply(int_mol)
#
#                         elif edit[0] == 'Delete Bond':
#                             edit_exe = BondEditAction(
#                                 *path['edits_atom'][i], *edit[1], action_vocab='Delete Bond')
#                             path['rxn_actions'].append(edit_exe)
#                             int_mol = edit_exe.apply(int_mol)
#
#                         if edit[0] == 'Change Bond':
#                             edit_exe = BondEditAction(
#                                 *path['edits_atom'][i], *edit[1], action_vocab='Change Bond')
#                             path['rxn_actions'].append(edit_exe)
#                             int_mol = edit_exe.apply(int_mol)
#
#                         if edit[0] == 'Attaching LG':
#                             edit_exe = AddGroupAction(
#                                 path['edits_atom'][i], edit[1], action_vocab='Attaching LG')
#                             path['rxn_actions'].append(edit_exe)
#                             int_mol = edit_exe.apply(int_mol)
#
#                     finished_paths.append(path)
#
#                 except Exception as e:
#                     print(f'Exception while final mol to Smiles: {str(e)}')
#                     path['final_smi'] = 'final_smi_unmapped'
#                     finished_paths.append(path)
#
#         return finished_paths
