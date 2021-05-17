import torch
from time import time
import dgl
import pickle
import pandas as pd
import os
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, RDKFingerprint, MACCSkeys
import logging
from pandarallel import pandarallel

logger = logging.getLogger()
RDLogger.DisableLog('rdApp.*')


class NormalMolecularGraph:
    def __init__(self,
                 dataset,
                 in_dim,
                 back_and_forth=True,
                 structure_aware=True,
                 smiles='clean_smiles'):
        pandarallel.initialize()
        self.dataset = dataset
        # self.file_path = f'../data/{dataset}/smiles.csv'
        self.proj_path = Path(__file__).parent.resolve().parent.resolve().parent.resolve()
        self.file_path = self.proj_path / 'data' / f'{dataset}' / f'{smiles}.csv'
        if not self.file_path.exists():
            self.file_path = self.proj_path / 'data' / f'{dataset}' / 'smiles.csv'
        self.in_dim = in_dim
        self.back_and_forth = back_and_forth
        self.structure_aware = structure_aware

        self.id2atom, self.atom2id = self.get_atom_info()
        if self.structure_aware:
            self.id2fg, self.fg2id = self.get_fg_info()

        tic = time()
        self.data = self.load_smiles()

        logger.info(
            f'Loading {self.dataset} totally consumed {time() - tic:.4f} seconds.')

    def mol_parallel_func(self, x):
        return Chem.MolFromSmiles(x)

    def load_smiles(self):
        start = time()
        data = pd.read_csv(self.file_path, header=0, usecols=['smiles'])
        print(f'load time: {time() - start:.5f}s')

        data['mol'] = data['smiles'].parallel_apply(self.mol_parallel_func)
        start = time()
        print('start mol2graph')
        data['graph'] = data['mol'].parallel_apply(self.mol2graph)
        print(f'applying graph consumed {time() - start:.5f}s')
        start = time()
        data['fp_morgan'] = data['mol'].parallel_apply(self.mol2morgan)
        data['fp_rdk'] = data['mol'].parallel_apply(self.mol2rdk)
        data['fp_macc'] = data['mol'].parallel_apply(self.mol2macc)

        print(f"applying fingerprint consumed {time() - start:.5f}s")
        return data

    def mol2fingerprint(self, mol):
        """
        choose a similarity function to generate fingerprint for each mol
        """
        if self.sim_function == 'morgan':
            fingerprint = AllChem.GetMorganFingerprint(mol, 2)
        elif self.sim_function == 'rdk':
            fingerprint = RDKFingerprint(mol)
        elif self.sim_function == 'macc':
            fingerprint = MACCSkeys.GenMACCSKeys(mol)
        return fingerprint

    def mol2morgan(self, mol):
        return AllChem.GetMorganFingerprint(mol, 2)

    def mol2rdk(self, mol):
        return RDKFingerprint(mol)

    def mol2macc(self, mol):
        return MACCSkeys.GenMACCSKeys(mol)

    def get_atom_info(self):
        """collect a global atom list"""
        # id2atom = [a for atom_list in self.data.atom_symbol for a in atom_list]
        # id2atom = sorted(list(set(id2atom)))
        # atom2id = {atom: idx for idx, atom in enumerate(id2atom)}

        atoms = []
        with open(self.proj_path / 'data' / 'atom.txt', encoding='utf-8') as f:
            for atom in f:
                atom = atom.strip()
                atoms.append(atom)
        id2atom = sorted(atoms)
        atom2id = {atom: idx for idx, atom in enumerate(id2atom)}
        return id2atom, atom2id

    def get_fg_info(self):
        """collect a global functional group list"""
        funcgroup = []
        with open(self.proj_path / 'data' / 'funcgroup.txt', encoding='utf-8') as f:
            for fg in f:
                fg = fg.strip()
                if fg not in funcgroup:
                    funcgroup.append(fg)
        # list of [fg_name, fg_mol], sorted from large mol to small mol
        id2fg = [(fg, Chem.MolFromSmarts(fg)) for fg in funcgroup]
        id2fg = sorted(id2fg, key=lambda x: x[1].GetNumAtoms(), reverse=True)
        fg2id = {fg[0]: idx for idx, fg in enumerate(id2fg)}
        return id2fg, fg2id

    def mol2atom_in_fg(self, mol):
        """
        find the name of corresponding functional group for each atom in mol.
        """
        fg_names = []
        fg_matches = []
        for fg_name, fg_mol in self.id2fg:
            atom_indices = mol.GetSubstructMatches(fg_mol)
            if atom_indices:
                fg_names.append(fg_name)
                # list of tuples of tuple like (4, 5, 6, 7)
                fg_matches.append(atom_indices)

        selected_fg = [fg_matches[0][0]]  # (4, 5, 6, 7)
        selected_atoms = []
        selected_atoms.extend(fg_matches[0][0])
        atom_in_fg_names = [fg_names[0]] * len(fg_matches[0][0])
        for fg_name, fg_match_tuple in zip(fg_names[1:], fg_matches[1:]):
            for fg_match in fg_match_tuple:
                if not set(fg_match) & set(selected_atoms):
                    selected_fg.append(fg_match)
                    selected_atoms.extend(fg_match)
                    atom_in_fg_names.extend([fg_name] * len(fg_match))

        return dict(zip(selected_atoms, atom_in_fg_names))

    def mol2graph(self, mol):
        # 0. Find connected atoms. Note that there are disconnected nodes in molecules.
        connected_atom_idx = []
        for bond in mol.GetBonds():
            idx_s = bond.GetBeginAtomIdx()
            idx_t = bond.GetEndAtomIdx()
            connected_atom_idx.append(idx_s)
            connected_atom_idx.append(idx_t)
        connected_atom_idx = sorted(list(set(connected_atom_idx)))
        # reset connected atoms
        connected_atom_idx = {k: v
                              for k, v in zip(connected_atom_idx, list(range(len(connected_atom_idx))))}

        # 1. build graph with basic structure info
        src_idx = []
        tgt_idx = []
        for bond in mol.GetBonds():
            idx_s = bond.GetBeginAtomIdx()
            idx_t = bond.GetEndAtomIdx()
            idx_s = connected_atom_idx[idx_s]
            idx_t = connected_atom_idx[idx_t]

            src_idx.append(idx_s)
            tgt_idx.append(idx_t)
            if self.back_and_forth:
                src_idx.append(idx_t)
                tgt_idx.append(idx_s)

        g = dgl.graph((src_idx, tgt_idx), idtype=torch.int32)

        # 2. add atom features and functional group features
        atoms_feat = []
        fgs_feat = []
        for atom in mol.GetAtoms():
            node_id = atom.GetIdx()
            # skip disconnected atoms
            if node_id not in connected_atom_idx:
                continue

            atom_name = atom.GetSymbol()  # 'C'
            atom_idx = self.atom2id[atom_name]
            # atom_feat = torch.tensor([atom_idx], dtype=torch.long).unsqueeze(0)
            atom_feat = [atom_idx]
            atoms_feat.append(atom_feat)

            if self.structure_aware:
                atom2fg_names = self.mol2atom_in_fg(mol)
                fg_name = atom2fg_names.get(node_id, "NOTFOUND")
                if fg_name == 'NOTFOUND':
                    # fg_feat = torch.tensor([0], dtype=torch.long).unsqueeze(0)
                    fg_feat = [0]
                else:
                    # map functional group name to global idx
                    fg_idx = self.fg2id[fg_name]
                    # fg_feat = torch.tensor([fg_idx + 1], dtype=torch.long).unsqueeze(0)
                    fg_feat = [fg_idx + 1]
                    # row indexing the eye matrix
                    # fg_one_hot = self.id2fg_one_hot[fg_idx].unsqueeze(0)
                fgs_feat.append(fg_feat)

        # g.ndata['atom_feat'] = torch.cat(atoms_feat, dim=0)
        g.ndata['atom_feat'] = torch.tensor(atoms_feat, dtype=torch.long)
        if self.structure_aware:
            # g.ndata['fg_feat'] = torch.cat(fgs_feat, dim=0)
            g.ndata['fg_feat'] = torch.tensor(fgs_feat, dtype=torch.long)
        return g


if __name__ == '__main__':
    tic = time()
    mg = NormalMolecularGraph(
        dataset='zinc',
        in_dim=512,
        structure_aware=True
    )
    print(time() - tic)
    # print(type(mg.data.fingerprint[0]))
    smile = "n1c2c(ccc1N)cccc2"
    mol = Chem.MolFromSmiles(smile)
    graph = mg.mol2graph(mol)
    print(graph)
    print(graph.ndata['atom_feat'])
    if 'fg_feat' in graph.ndata:
        print(graph.ndata['fg_feat'])