import mdtraj as md
import yaml
import pandas as pd
import os

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
            "catalytic_monomer": [408, 812],
            "ACP": [813, 886],
            "ref": None,
            "output": None,
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
        if np.shape(self.catalytic_monomer) != (1,2):
            raise ParameterError("Catalytic Monomer residue range should be a 1 by 2 list or array")
        if np.shape(self.ACP) != (1,2):
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
                
        params_int = ['chain_length']
        for i in params_int:
            if type(getattr(self, i)) != int:
                raise ParameterError(f"The parameter '{i}' should be an integer.")
        
        params_str = ['gro', 'traj', 'file_path']
        for i in params_str:
            if type(getattr(self, i)) != str:
                raise ParameterError(f"The parameter '{i}' should be a string.")
    
    def align(self, i, n):
        ref_struct = md.load(self.ref[n][i])
        atom_indices = traj.topology.select(f'resid > {self.catalytic_monomer[0] - 2} and resid < {self.catalytic_monomer[1]}')
        ref_atom_indices = ref_struct.topology.select(f'resid > 406 and resid < 812')
        traj_align = traj.superpose(ref_struct, atom_indices, ref_atom_indices)

        return traj_align, ref_struct
    
    def chain_rmsd(self, traj, ref):
        chain_atom_name = ['C12', 'C13', 'C14', 'C15']
        for i in range(self.chain_length - 4):
            chain_atom_name.append(f'C{i+16}')
        print(f'atom names: {chain_atom_name}')

        traj_atom_num, ref_atom_num = [], []
        for name in chain_atom_name:
            traj_atom_num.append(traj.topology.select(f'name {name}'))
            ref_atom_num.append(ref.topology.select(f'name {name}'))
        
        rmsd = []
        for t in traj.n_frames:
            tot_square_dist = 0
            for tn, rn in zip(traj_atom_num, ref_atom_num):
                tot_square_dist += (traj.xyz[tn, t] - ref.xyz[rn, t])**2
            rmsd.append(np.sqrt(tot_square_dist/len(chain_atom_name)))

        return rmsd
parser = argparse.ArgumentParser(description='This script runs pocket determination for FabB and FabF.')
parser.add_argument('-y', '--yaml', type=str, default='params.yaml', help='The file path of the input YAML file that contains REXEE parameters. (Default: params.yaml)')

DP = DeterminePocket(arg.yaml)

# Loop through all structures and references
for n in len(DP.gro):
    # Step 1: Load trajectory
    traj = md.load(f'{DP.file_path}/{DP.traj[n]}', top=f'{DP.file_path}/{DP.gro[n]}')

    rmsd_to_all_ref = np.zeros((len(DP.ref), len(traj.n_frames)))
    for ref in len(DP.ref):
        # Step 2: Align to reference structure on catalytic monomer
        traj_align, ref_struct = DP.align(n, ref)

        # Step 3: Compute the RMSD for the atoms within the acyl chain for 
        rmsd_to_all_ref[ref, :] = DP.chain_rmsd(traj_align, ref_struct)
    df = pd.DataFrame({'RMSD A': rmsd_to_all_ref[0,:], 'RMSD B': rmsd_to_all_ref[1,:], 'RMSD C': rmsd_to_all_ref[2,:]})
    df.to_csv('test.csv')
    exit()
