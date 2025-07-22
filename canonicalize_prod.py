"""
Canonicalize the product SMILES, and then use substructure matching to infer
the correspondence to the original atom-mapped order. This correspondence is then
used to renumber the reactant atoms.
"""


from rdkit import Chem
import os
import argparse
import pandas as pd


def canonicalize_prod(p, mapped_rxn):
    import copy
    p = copy.deepcopy(p)
    p = canonicalize(p)
    p_mol = Chem.MolFromSmiles(p)

    if p_mol is not None:
        for atom in p_mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)
        p = Chem.MolToSmiles(p_mol)
    else:
        print("Error: p_mol is None")
        print("mapped_rxn:", mapped_rxn)
        # for atom in p_mol.GetAtoms():
    #     atom.SetAtomMapNum(atom.GetIdx() + 1)
    # p = Chem.MolToSmiles(p_mol)
    return p


def canonicalize(smiles):
    try:
        tmp = Chem.MolFromSmiles(smiles)
    except:
        print('no mol', flush=True)
        return smiles
    if tmp is None:
        return smiles
    tmp = Chem.RemoveHs(tmp)
    [a.ClearProp('molAtomMapNumber') for a in tmp.GetAtoms()]
    return Chem.MolToSmiles(tmp)


def fix_charge(mol):
    # fix simple atomic charge, eg. 'COO-', 'CH3O-', '(S=O)O-', '-NH3+', 'NH4+', 'NH2+', 'S-'
    for atom in mol.GetAtoms():
        explicit_hs = atom.GetNumExplicitHs()
        charge = atom.GetFormalCharge()
        bond_vals = int(sum([b.GetBondTypeAsDouble()
                        for b in atom.GetBonds()]))
        if atom.GetSymbol() == 'O' and bond_vals == 1 and charge == -1 and explicit_hs == 0:
            if atom.GetNeighbors()[0].GetSymbol() != 'N':
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(1)

        if atom.GetSymbol() == 'N' and bond_vals == 1 and charge == 1 and explicit_hs == 3:
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(2)

        if atom.GetSymbol() == 'N' and bond_vals == 0 and charge == 1 and explicit_hs == 4:
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(3)

        if atom.GetSymbol() == 'N' and bond_vals == 2 and charge == 1 and explicit_hs == 2:
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(1)

        if atom.GetSymbol() == 'S' and charge == -1 and explicit_hs == 0 and bond_vals == 1:
            atom.SetNumExplicitHs(1)
            atom.SetFormalCharge(0)
    return mol


def infer_correspondence(p, mapped_rxn):
    orig_mol = Chem.MolFromSmiles(p)
    canon_mol = Chem.MolFromSmiles(canonicalize_prod(p, mapped_rxn))
    if canon_mol is None:
        print("Error: canon_mol is None")
        print("mapped_rxn:", mapped_rxn)  # Print the mapped_rxn content when canon_mol is None
        return {}
    matches = list(canon_mol.GetSubstructMatches(orig_mol))
    idx_amap = {atom.GetIdx(): atom.GetAtomMapNum()
                for atom in orig_mol.GetAtoms()}

    correspondence = {}
    if matches:
        for idx, match_idx in enumerate(matches[0]):
            match_anum = canon_mol.GetAtomWithIdx(match_idx).GetAtomMapNum()
            old_anum = idx_amap[idx]
            correspondence[old_anum] = match_anum
    return correspondence


def remap_rxn_smi(rxn_smi, mapped_rxn):
    r, p = rxn_smi.split(">>")
    canon_mol = Chem.MolFromSmiles(canonicalize_prod(p, mapped_rxn))
    correspondence = infer_correspondence(p, mapped_rxn)

    rmol = Chem.MolFromSmiles(r)
    if rmol is None or rmol.GetNumAtoms() <= 1:
        return rxn_smi, None

    for atom in rmol.GetAtoms():
        atomnum = atom.GetAtomMapNum()
        if atomnum in correspondence:
            newatomnum = correspondence[atomnum]
            atom.SetAtomMapNum(newatomnum)

    max_amap = max([atom.GetAtomMapNum() for atom in rmol.GetAtoms()])
    for atom in rmol.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            atom.SetAtomMapNum(max_amap + 1)
            max_amap += 1

    # fix simple atomic charge, eg. 'COO-', 'CH3O-', '(S=O)O-', '-NH3+', 'NH4+', 'NH2+', 'S-'
    rmol = fix_charge(rmol)
    canon_mol = fix_charge(canon_mol)
    if rmol is None:
        raise ValueError("rmol is None after fix_charge")
    if canon_mol is None:
        raise ValueError("canon_mol is None after fix_charge")

    rmol_smiles = Chem.MolToSmiles(rmol)
    canon_mol_smiles = Chem.MolToSmiles(canon_mol)
    # rmol = Chem.MolFromSmiles(Chem.MolToSmiles(rmol))
    rxn_smi_new = Chem.MolToSmiles(rmol) + ">>" + Chem.MolToSmiles(canon_mol)
    return rxn_smi_new, correspondence

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='C-H Arylation',
                        help='dataset: C-H Arylation or USPTO_full')
    parser.add_argument('--mode', type=str, default='valid-1',
                        help='Type of dataset being prepared: train-1 or valid-1 or test-1')
    args = parser.parse_args()

    args.dataset = args.dataset.lower()
    datadir = f'data/{args.dataset}/'
    new_file = f'canonicalized_{args.mode}.csv'
    filename = f'raw_{args.mode}.csv'
    df = pd.read_csv(os.path.join(datadir, filename))
    print(f"Processing file of size: {len(df)}")

    if args.dataset == 'C-H Arylation':
        new_dict = {'id': [], 'class': [], 'mapped_rxn': []}
    else:
        new_dict = {'id': [], 'mapped_rxn': []}
    for idx in range(len(df)):
        element = df.loc[idx]
        if args.dataset == 'C-H Arylation':
            uspto_id, class_id, rxn_smi = element['id'], element['class'], element['mapped_rxn']
        else:
            uspto_id, rxn_smi = element['id'], element['mapped_rxn']

        rxn_smi_new, _ = remap_rxn_smi(rxn_smi, element['mapped_rxn'])
        new_dict['id'].append(uspto_id)
        if args.dataset == 'C-H Arylation':
            new_dict['class'].append(class_id)
        new_dict['mapped_rxn'].append(rxn_smi_new)

    new_df = pd.DataFrame.from_dict(new_dict)
    new_df.to_csv(os.path.join(datadir, new_file), index=False)


if __name__ == "__main__":
    main()
