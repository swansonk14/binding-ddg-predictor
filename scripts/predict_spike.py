"""Predicts change in binding energy for multiple PDB files across all RBD mutations."""
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from copy import deepcopy
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List

import pandas as pd
import torch
from pyrosetta.io import pose_from_pdb
from pyrosetta.rosetta.core.io.pdb import dump_pdb
from pyrosetta.rosetta.core.pose import get_chain_from_chain_id, get_chains, Pose
from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
from pyrosetta.toolbox.mutants import mutate_residue
from tqdm import tqdm

from models.predictor import DDGPredictor
from utils.misc import *
from utils.data import *
from utils.protein import *


AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
AMINO_ACID_3_to_1 = {
    'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
    'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
    'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'
}
REPACK_DISTANCE = 8.0
# TODO: different score function?
DDG_SCORE_FUNCTION = ScoreFunctionFactory.create_score_function('ddg')


def get_spike_chain(pdb_path: Path) -> str:
    """Gets the letter of the chain containing the spike protein.

    :param pdb_path: Path to a PDB file containing a protein complex with a spike chain.
    :return: The letter of the chain containing the spike protein.
    """
    # Load protein info from PDB file
    structure = PDBParser().get_structure(pdb_path.stem, pdb_path)

    # Determine the letter of the chain with the spike protein
    spike_chain = None
    for chain in structure.header['compound'].values():
        if 'spike' in chain['molecule'].lower():
            if spike_chain is not None:
                raise ValueError('Found two spike chains.')
            else:
                spike_chain = chain['chain'].upper()

    if spike_chain is None:
        raise ValueError('Could not find spike chain.')

    return spike_chain


def predict_mutation(wildtype_pdb_path: str,
                     mutation_pdb_path: str,
                     model: DDGPredictor,
                     device: str) -> float:
    """Predicts DDG for a single mutation."""
    batch = load_wt_mut_pdb_pair(wildtype_pdb_path, mutation_pdb_path)
    batch = recursive_to(batch, device)

    with torch.no_grad():
        pred = model(batch['wt'], batch['mut']).item()

    return pred


def predict_residue(pose: Pose,
                    residue: int,
                    model: DDGPredictor,
                    device: str) -> List[float]:
    """Predicts DDG for a single residue across all mutations."""
    # Get wildtype amino acid
    wildtype_amino_acid = AMINO_ACID_3_to_1[pose.residue(residue).name()[:3]]

    # Create temporary PDB files
    wildtype_pdb_file = NamedTemporaryFile(suffix='.pdb')
    mutation_pdb_file = NamedTemporaryFile(suffix='.pdb')

    # Repack wildtype pose
    mutate_residue(pose, residue, wildtype_amino_acid, REPACK_DISTANCE, DDG_SCORE_FUNCTION)

    # Save wildtype pose to PDB file
    dump_pdb(pose, wildtype_pdb_file.name)

    # Get mutant amino acids
    mutant_amino_acids = deepcopy(AMINO_ACIDS)
    mutant_amino_acids.remove(wildtype_amino_acid)

    # Loop through amino acids and predict DDG
    ddg = []

    for index, amino_acid in enumerate(AMINO_ACIDS):
        if amino_acid == wildtype_amino_acid:
            ddg.append(0.0)
        else:
            # Mutate and repack pose with mutation
            mutate_residue(pose, residue, amino_acid, REPACK_DISTANCE, DDG_SCORE_FUNCTION)

            # Save mutation pose to PDB file
            dump_pdb(pose, mutation_pdb_file.name)

            # Predict DDG
            ddg.append(predict_mutation(
                wildtype_pdb_path=wildtype_pdb_file.name,
                mutation_pdb_path=mutation_pdb_file.name,
                model=model,
                device=device
            ))

    # Close temporary files
    wildtype_pdb_file.close()
    mutation_pdb_file.close()

    return ddg


def predict_protein(pdb_path: Path,
                    save_path: Path,
                    model: DDGPredictor,
                    device: str) -> None:
    """Predicts DDG for a single PDB file across all RBD mutations."""
    # Load protein pose from PDB file
    pose: Pose = pose_from_pdb(str(pdb_path))

    # Get spike chain
    spike_chain_letter = get_spike_chain(pdb_path)
    spike_chains = [
        chain_id
        for chain_id in get_chains(pose)
        if get_chain_from_chain_id(chain_id, pose) == spike_chain_letter and
           pose.chain_end(chain_id) - pose.chain_begin(chain_id) > 0
    ]

    if len(spike_chains) > 1:
        raise ValueError('Multiple spike chains.')

    spike_chain = spike_chains[0]

    # Get spike residues
    spike_residues = list(range(pose.chain_begin(spike_chain), pose.chain_end(spike_chain) + 1))

    # Get wildtype residue letters
    wildtype_residues = [
        AMINO_ACID_3_to_1[pose.residue(residue).name()[:3]] for residue in spike_residues
    ]

    # Predict DDG for each residue in the spike chain
    ddg_matrix = [
        predict_residue(
            pose=pose,
            residue=residue,
            model=model,
            device=device
        )
        for residue in tqdm(spike_residues, leave=False)
    ]

    # Save delta delta G data
    data = pd.DataFrame(data=ddg_matrix, index=wildtype_residues, columns=AMINO_ACIDS)
    data.to_csv(save_path)


def predict_multi_protein(data_dir: Path,
                          save_dir: Path,
                          model: str,
                          device: str) -> None:
    """Predicts DDG for multiple PDB files across all RBD mutations."""
    # Set up PDB and save paths
    pdb_paths = sorted(data_dir.glob('*.pdb'))

    # Load model
    ckpt = torch.load(model)
    config = ckpt['config']
    weight = ckpt['model']
    model = DDGPredictor(config.model).to(device)
    model.load_state_dict(weight)
    model.eval()

    # Predict DDG for each protein complex
    for pdb_path in tqdm(pdb_paths):
        predict_protein(
            pdb_path=pdb_path,
            save_path=save_dir / pdb_path.with_suffix('.csv').name,
            model=model,
            device=device
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, default='../data/cao_omicron/pdb/pdb')
    parser.add_argument('--save_dir', type=Path, default='../preds/cao_omicron/binding-ddg-predictor')
    parser.add_argument('--model', type=str, default='data/model.pt')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    predict_multi_protein(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        model=args.model,
        device=args.device
    )
