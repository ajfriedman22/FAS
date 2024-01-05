import mdtraj as md
import numpy as np
import argparse
from itertools import product
import pandas as pd
from tqdm import tqdm

# Declare arguments
parser = argparse.ArgumentParser(description = 'Determination of Different Binding Complex Measures')
parser.add_argument('-t', required=True, help='Array of file names for input trajectory')
parser.add_argument('-g', required=True, help= 'Array of file names for input topology (gro format)')

#Import Arguments
args = parser.parse_args()
File_traj_all = args.t
File_gro_all = args.g
miss_res = args.m
lig = args.l
lig_name = args.ln
sect = args.sect
if sect.split('.')[-1] != 'txt' and sect != 'none': #Add default file extension if not in input
    sect = sect + '.txt'

#Load reference PDB
ref = md.load('FabG_ACoA_bac.pdb')

#Declare empty arrays
contact_per_R172 = np.zeros(len(File_traj_all))
contact_per_R129 = np.zeros(len(File_traj_all))
C1_dist_S138 = np.zeros(len(File_traj_all))
C1_dist_Y151 = np.zeros(len(File_traj_all))
C1_dist_K155 = np.zeros(len(File_traj_all))
RMSD_CoA_C1 = np.zeros(len(File_traj_all))
RMSD_CoA_C2 = np.zeros(len(File_traj_all))
RMSD_CoA_C3 = np.zeros(len(File_traj_all))
RMSD_CoA_C4 = np.zeros(len(File_traj_all))

#Loop through all input files
for i in tqdm(range(len(File_traj_all))):
    #Load Trajectory
    if File_traj_all[I].split('.')[-1] != 'xtc': #Add file extension if not in input
        File_traj_all[i] = File_traj_all[i] + '.xtc'    
    if File_gro_all[i].split('.')[-1] != 'gro': #Add default file extension if not in input
        File_gro_all[i] = File_gro_all[i] + '.gro'

    traj = md.load(File_traj_all[i], top=File_gro_all[i])
    top = traj.topology

    #Determine residue contact between ACP and R172 and R129
    ACP = np.linspace(243, 320, num=77)
    pairs = list(product(ACP, [171, 128]))
    dist = md.compute_contacts(traj, contacts=pairs, scheme='closest-heavy')

    frames, pairs = np.shape(dist)
    count_R172, count_R129 = 0,0
    for t in range(frames):
        for p in range(pairs):
            if dist[t][p] < 0.05:
                if p < 77:
                    count_R172 += 1
                else:
                    count_R129 += 1
    contact_per_R172[i] = 100 * (count_R172/frames)
    contact_per_R129[i] = 100 * (count_R129/frames)

    #Determine distance between C1 and res S138, Y151, and K155
    C1 = top.select('resname C4 and name C1')
    res = ['137', '150', '154']
    array = ['S138', 'Y151', 'K155']
    for r in range(len(res)):
        res_sel = top.select(f'resid {res[r]}')
        atom_pairs = list(product(C1, res_sel))

        dist = md.compute_distances(traj, atom_pairs=atom_pairs)

        mean_dist = np.mean(dist, axis=0)
        print(len(mean_dist))

        globals()['C1_dist_%s' % array[r]] = min(mean_dist)
    
    #RMSD between acetyl CoA and acyl chain
    atom = ['C1', 'C2', 'C3', 'C4']
    align_index = top.select('backbone and resid < 240')
    ref_align = top.select('backbone and resid > 2 resid < 242')
    traj_align = md.superpose(traj, ref, atom_indices=align_index, ref_atom_indices=ref_align)#align reference structure
    for a in atom:
        atom_index = top.select('resname C4 and name {a}')
        ref_index = ref.topology.select('resname CAA and name {a}')

        rmsd = md.rmsd(traj_align, ref, atom_indices=atom_index, ref_atom_indices=ref_index)
        globals()['RMSD_CoA_%s' % a] = rmsd
df = pd.DataFrame({'% Contact w/ R172': contact_per_R172, '% Contact w/ R129': contact_per_R129, 'Minimum Mean Distance b/w C1 and S138': C1_dist_S138, 
                  'Minimum Mean Distance b/w C1 and Y151': C1_dist_Y151, 'Minimum Mean Distance b/w C1 and K155': C1_dist_K155, 'Relative RMSD for C1': RMSD_CoA_C1, 'Relative RMSD for C2': RMSD_CoA_C2,
                  'Relative RMSD for C3': RMSD_CoA_C3, 'Relative RMSD for C4': RMSD_CoA_C4})
df.to_csv('Compare_FabG_bind.csv')