import mdtraj as md
import yaml
import pandas as pd
import os
import numpy as np
import argparse
from tqdm import tqdm

class ParameterError(Exception):
    """Error raised when detecting improperly specified parameters in the YAML file."""

class DeterminePocket:
    """
    A class that provides a variety of functions for computing the correct pocket
    All parameters in the input YAML file will be assigned to an
    attribute in the class. (All the the attributes are assigned by :obj:`set_params`
    except that :code:`yaml` is assigned by :code:`__init__`.)
    """

    def __init__(self, yaml_file):
        self.yaml = yaml_file
        self.set_params()

    def set_params(self):
        """
        Sets up or reads in the user-defined parameters from an input YAML file and an MDP template.
        This function is called to instantiate the class in the :code:`__init__` function of
        class.
        """
        self.warnings = []  # Store warnings, if any.

        # Step 1: Read in parameters from the YAML file.
        with open(self.yaml) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        for attr in params:
            setattr(self, attr, params[attr])

        # Step 2: Handle the compulsory YAML parameters
        required_args = [
            "gro",
            "traj",
            "chain_length",
            "name",
        ]
        for i in required_args:
            if hasattr(self, i) is False or getattr(self, i) is None:
                raise ParameterError(
                    f"Required parameter '{i}' not specified in {self.yaml}."
                )  # noqa: F405

        # Step 3: Handle the optional YAML parameters
        # Key: Optional argument; Value: Default value
        optional_args = {
            "file_path": './',
            "catalytic_monomer": [406, 811],
            "ACP": [812, 886],
            "ref": None,
            "output": None,
            "ref_residue_range": [406, 811],
            "output_all_rmsd": "all_rmsd.csv",
            "output_pocket_per_frame": "pocket_per_frame.csv",
            "output_pocket_summary": "percent_occupancy.csv",
        }

        for i in optional_args:
            if hasattr(self, i) is False or getattr(self, i) is None:
                setattr(self, i, optional_args[i])

        # all_args: Arguments that can be specified in the YAML file.
        all_args = required_args + list(optional_args.keys())
        for i in params:
            if i not in all_args:
                self.warnings.append(f'Warning: Parameter "{i}" specified in the input YAML file is not recognizable.')

        # Step 4: Check if the parameters in the YAML file are well-defined
        if len(self.gro) != len(self.traj):
            raise ParameterError("Length of trajectory and gro lists should be equal")
        if len(self.gro) != len(self.chain_length):
            raise ParameterError("Length of trajectory and chain length lists should be equal")
        if np.shape(self.catalytic_monomer) != (2,):
            raise ParameterError(f"Catalytic Monomer residue range should be a 1 by 2 list or array not {np.shape(self.catalytic_monomer)}")
        if np.shape(self.ACP) != (2,):
            raise ParameterError("ACP residue range should be a 1 by 2 list or array")
        if self.ref is not None and len(self.ref) != len(self.gro):
            raise ParameterError("If provided the reference list should be the same length as the input trajectories")
        if self.ref is None:
            ref_list = []
            script_path = os.path.abspath(__file__)
            script_directory = os.path.dirname(script_path)
            for l in self.chain_length:
                if l not in [4, 6, 8, 10, 12, 14, 16]:
                    raise ParameterError(f"Invalid chain length of {l}")
                ref_list.append([f'{script_directory}/reference_structs/C{l}_A.pdb', f'{script_directory}/reference_structs/C{l}_B.pdb', f'{script_directory}/reference_structs/C{l}_C.pdb'])
            self.ref = ref_list
                
        params_int = ['chain_length']
        for i in params_int:
            if type(getattr(self, i)[0]) != int:
                raise ParameterError(f"The parameter '{i}' should be an integer.")
        
        params_str_list = ['gro', 'traj', 'name']
        for i in params_str_list:
            if type(getattr(self, i)[0]) != str:
                raise ParameterError(f"The parameter '{i}' should be a string.")
            
        params_str = ['file_path', "output_all_rmsd", "output_pocket_per_frame", "output_pocket_summary"]
        for i in params_str:
            if type(getattr(self, i)) != str:
                raise ParameterError(f"The parameter '{i}' should be a string.")
    
    def align(self, i, n):
        ref_struct = md.load(self.ref[n][i])
        atom_indices = traj.topology.select(f'resid > {self.catalytic_monomer[0] - 2} and resid < {self.catalytic_monomer[1]} and backbone')
        ref_atom_indices = ref_struct.topology.select(f'resid > {self.ref_residue_range[0] - 2} and resid < {self.ref_residue_range[1]} and backbone')
        traj_align = traj.superpose(reference = ref_struct, atom_indices = atom_indices, ref_atom_indices = ref_atom_indices)

        return traj_align, ref_struct
    
    def chain_rmsd(self, traj, ref, n):
        chain_atom_name = ['C12', 'C13', 'C14', 'C15']
        for i in range(self.chain_length[n] - 4):
            chain_atom_name.append(f'C{i+16}')

        traj_atom_num, ref_atom_num = [], []
        for name in chain_atom_name:
            traj_atom_num.append(traj.topology.select(f'name {name}'))
            ref_atom_num.append(ref.topology.select(f'name {name}'))

        rmsd = []
        for t in range(traj.n_frames):
            tot_square_dist = 0
            for tn, rn in zip(traj_atom_num, ref_atom_num):
                tn = tn[0]
                rn = rn[0]
                dist = np.sqrt((traj.xyz[t][tn][0] - ref.xyz[0][rn][0])**2 + (traj.xyz[t][tn][1] - ref.xyz[0][rn][1])**2 + (traj.xyz[t][tn][2] - ref.xyz[0][rn][2])**2)
                tot_square_dist += dist
            rmsd.append(np.sqrt(tot_square_dist/len(chain_atom_name)))

        return rmsd
parser = argparse.ArgumentParser(description='This script runs pocket determination for FabB and FabF.')
parser.add_argument('-y', '--yaml', type=str, default='params.yaml', help='The file path of the input YAML file that contains REXEE parameters. (Default: params.yaml)')

args = parser.parse_args()

DP = DeterminePocket(args.yaml)

# Save output files
all_rmsd_df = pd.DataFrame()
pocket_per_frame_df = pd.DataFrame()
perA, perB, perC = np.zeros(len(DP.gro)), np.zeros(len(DP.gro)), np.zeros(len(DP.gro))
# Loop through all structures and references
for n in tqdm(range(len(DP.gro))):
    # Step 1: Load trajectory
    traj = md.load(f'{DP.file_path}/{DP.traj[n]}', top=f'{DP.file_path}/{DP.gro[n]}')

    rmsd_to_all_ref = np.zeros((len(DP.ref[0]), traj.n_frames))
    for ref in range(len(DP.ref[0])):
        # Step 2: Align to reference structure on catalytic monomer
        traj_align, ref_struct = DP.align(ref, n)

        # Step 3: Compute the RMSD for the atoms within the acyl chain for 
        rmsd_to_all_ref[ref, :] = DP.chain_rmsd(traj_align, ref_struct, n)
    df = pd.DataFrame({'Name': DP.name[n], 'RMSD A': rmsd_to_all_ref[0,:], 'RMSD B': rmsd_to_all_ref[1,:], 'RMSD C': rmsd_to_all_ref[2,:]})
    all_rmsd_df = pd.concat([all_rmsd_df, df])

    # Step 4: Classify pockets
    pocket = []
    pocketA, pocketB, pocketC = 0, 0, 0
    for t in range(traj.n_frames):
        if rmsd_to_all_ref[0,t] < 0.85 and rmsd_to_all_ref[0,t] < rmsd_to_all_ref[1,t] and rmsd_to_all_ref[0,t] < rmsd_to_all_ref[2,t]:
            pocket.append('A')
            pocketA += 1
        elif rmsd_to_all_ref[1,t] < 0.8 and rmsd_to_all_ref[1,t] < rmsd_to_all_ref[0,t] and rmsd_to_all_ref[1,t] < rmsd_to_all_ref[2,t]:
            pocket.append('B')
            pocketB += 1
        elif rmsd_to_all_ref[2,t] < 0.8 and rmsd_to_all_ref[2,t] < rmsd_to_all_ref[1,t] and rmsd_to_all_ref[2,t] < rmsd_to_all_ref[1,t]:
            pocket.append('C')
            pocketC += 1
        else:
            pocket.append('D')
    df = pd.DataFrame({'Name': DP.name[n], 'Frame': np.arange(0, traj.n_frames, step=1), 'Pocket': pocket})
    pocket_per_frame_df = pd.concat([pocket_per_frame_df, df])
    
    perA[n] = 100*pocketA/traj.n_frames
    perB[n] = 100*pocketB/traj.n_frames
    perC[n] = 100*pocketC/traj.n_frames

pocket_summary_df = pd.DataFrame({'Name': DP.name, 'Percent Pocket A': perA, 'Percent Pocket B': perB, 'Percent Pocket C': perC})

all_rmsd_df.to_csv(DP.output_all_rmsd)
pocket_per_frame_df.to_csv(DP.output_pocket_per_frame)
pocket_summary_df.to_csv(DP.output_pocket_summary)