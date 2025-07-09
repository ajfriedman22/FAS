import mdtraj as md
import yaml
import pandas as pd
import os
import numpy as np
import argparse
from tqdm import tqdm
import itertools

class ParameterError(Exception):
    """Error raised when detecting improperly specified parameters in the YAML file."""

class DetermineInteractions:
    """
    A class that provides a variety of functions for computing the interactions 
    between residues based on the pocket in each frame.
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
            "interaction_residues",
            "interaction_name",
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
            "output_inter_dist": "inter_dist.csv",
            "output_inter_per": "inter_per.csv",
            "output_top_inters": "inter_top.csv",
            "output_cat_comp": "per_cat_comp.csv",
            "top_inter_per": 75,
            "pockets_input": "pocket_per_frame.csv",
            "ignore_pockets": ['D'],
            "cat_define": [[164, 'SG'], [845, 'C12']],
            "cat_cutoff": 0.5,
            "contact_cutoff": 0.4,
            "enforce_cat_comp": True,
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
                
        params_int = ['top_inter_per']
        for i in params_int:
            if type(getattr(self, i)) != int:
                raise ParameterError(f"The parameter '{i}' should be an integer.")
        
        params_float = ['cat_cutoff', "contact_cutoff"]
        for i in params_float:
            if type(getattr(self, i)) != float:
                raise ParameterError(f"The parameter '{i}' should be a float.")

        params_str_list = ['gro', 'traj', 'name', 'ignore_pockets']
        for i in params_str_list:
            if type(getattr(self, i)[0]) != str:
                raise ParameterError(f"The parameter '{i}' should be a string.")
            
        params_str = ['file_path', "output_inter_dist", "output_inter_per", "output_top_inters", "output_cat_comp", "pockets_input"]
        for i in params_str:
            if type(getattr(self, i)) != str and type(getattr(self, i)) != type(None):
                raise ParameterError(f"The parameter '{i}' should be a string.")

parser = argparse.ArgumentParser(description='This script runs pocket determination for FabB and FabF.')
parser.add_argument('-y', '--yaml', type=str, default='params.yaml', help='The file path of the input YAML file that contains REXEE parameters. (Default: params.yaml)')

args = parser.parse_args()

DI = DetermineInteractions(args.yaml)

pockets_df = pd.read_csv(DI.pockets_input)
cat_comp_per = np.zeros(len(DI.gro))
cat_comp_per_by_pocket = np.zeros((len(DI.gro), 3))
dist_df = pd.DataFrame()
inter_per_df = pd.DataFrame()
inter_top_df = pd.DataFrame()

for n in tqdm(range(len(DI.gro))):
    # Step 1: Load trajectory
    traj = md.load(f'{DI.file_path}/{DI.traj[n]}', top=f'{DI.file_path}/{DI.gro[n]}')
    traj = traj[::20]

    # Step 2: Determine which frames are catalytically competant
    res1, res2 = DI.cat_define
    res1_num, res1_name = res1
    res2_num, res2_name = res2
    cat1 = traj.topology.select(f'resid {res1_num - 1} and name {res1_name}')[0]
    cat2 = traj.topology.select(f'resid {res2_num - 1} and name {res2_name}')[0]

    cat_dist = md.compute_distances(traj, [[cat1, cat2]])

    frame_cat = []
    for f, dist in enumerate(cat_dist):
        if dist < DI.cat_cutoff:
            frame_cat.append(f)
    cat_comp_per[n] = 100*len(frame_cat)/traj.n_frames

    # Step 3: Determine which pockets to process
    name = DI.name[n]
    pocket_options = list(pockets_df[pockets_df['Name'] == name]['Pocket'].sort_values().unique())
    pocket_list = pockets_df[pockets_df['Name'] == name]['Pocket'].to_list()[::20]
    for pocket in DI.ignore_pockets:
        if pocket in pocket_options:
            pocket_options.remove(pocket)

    # Step 4: Determine the catalytic competance by pocket
    for p, pocket in enumerate(pocket_options):
        p_count, c_count = 0, 0
        for f, dist in enumerate(cat_dist):
            if pocket_list[f] == pocket:
                p_count+=1
                if dist < DI.cat_cutoff:
                    c_count += 1
        if p < 3:
            cat_comp_per_by_pocket[n][p] = (100*c_count/p_count)

    # Step 5: Reduce trajectory to catalytically competatant frames only
    traj_cat = traj.slice(frame_cat)
    del traj
    pocket_cat = []
    for i in frame_cat:
        pocket_cat.append(pocket_list[i])

    # Step 6: Loop through residue contact pairs
    for d, pair in enumerate(DI.interaction_residues):
        # Step 7: Determine the residue pairs to compute contacts for
        set1, set2 = pair
        if set1[0] == set1[1]:
            res1 = [set1[0] - 1]
        else:
            res1 = np.arange(set1[0], set1[1])
        if set2[0] == set2[1]:
            res2 = [set2[0] - 1]
        else:
            res2 = np.arange(set2[0], set2[1])
        res_pairs = list(itertools.product(res1, res2))

        # Step 8: Compute contacts between all residue pairs
        dist, pairs = md.compute_contacts(traj_cat, contacts=res_pairs, scheme='closest-heavy', ignore_nonprotein=False, periodic=True)
        for i in range(len(res_pairs)):
            df = pd.DataFrame({'Name': name[n], 'Pocket': pocket, 'Interation Name': DI.interaction_name[d], 'Residue 1': pairs[i,0], 'Residue 2': pairs[i,1], 'Distance': dist[:,i], 'Pocket': pocket_cat})
            dist_df = pd.concat([dist_df, df])
        
        # Step 9: Loop through each pocket
        for pocket in pocket_options:
            # Step 10: Compute the percent time contacts are formed
            per_all = []
            for i in range(len(res_pairs)):
                pocket, contact = 0, 0
                for t in range(traj_cat.n_frames):
                    if pocket_list[t] == pocket:
                        pocket += 1
                        if dist[i, t] < DI.contact_cutoff:
                            contact += 1    
                per_all.append(100*contact/pocket)
            
            # Step 11: Update saved files
            df = pd.DataFrame({'Name': name[n], 'Pocket': pocket, 'Interaction Name': DI.interaction_name[d], 'Residue 1': pairs[:,0], 'Residue 2': pairs[:,1], 'Percent Interaction': per_all})
            select_df = df[df['Percent Interaction'] > 75]
            inter_per_df = pd.contact([inter_per_df, df])
            inter_top_df = pd.contact([inter_top_df, select_df])
    del traj_cat
# Step 12: Save output files
cat_comp_df = pd.DataFrame({'Name': DI.name, 'Total Percent Catalytically Competant': cat_comp_per, 'Percent Catalytically Competant in Pocket A': cat_comp_per_by_pocket[0,:], 'Percent Catalytically Competant in Pocket B': cat_comp_per_by_pocket[1,:], 'Percent Catalytically Competant in Pocket C': cat_comp_per_by_pocket[2,:]})
cat_comp_df.to_csv(DI.output_cat_comp)

dist_df.to_csv(DI.output_inter_dist)
inter_per_df.to_csv(DI.output_inter_per)
inter_top_df.to_csv(DI.output_top_inters)