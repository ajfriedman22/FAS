import random
import mdtraj as md
import argparse
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def conformer_sort(chain_pdb, current_sect):
    #Load_pdb
    chain = md.load(f'{chain_pdb}/{current_sect}_conformers.pdb')

    #Compute pairwise distances
    distances = np.empty((chain.n_frames, chain.n_frames))
    for i in range(chain.n_frames):
        distances[i] = md.rmsd(chain, chain, i)
    print('Max pairwise rmsd for ' + current_sect + ': %f nm' % np.max(distances))

    #Perform Clustering
    reduced_distances = squareform(distances, checks=False)
    link = linkage(reduced_distances, method='average') #The hierarchical clustering encoded as a matrix
    frame_list = dendrogram(link, no_labels=False, count_sort='descendent')['leaves']
    frame_cat = dendrogram(link, no_labels=False, count_sort='descendent')['leaves_color_list']

    fig = plt.figure()
    plt.title('RMSD Average linkage hierarchical clustering')
    _ = dendrogram(link, no_labels=True, count_sort='descendent')
    plt.savefig(f'dendrogram_{current_sect}.png')

    #Keep only one file per cluster
    frames_indv, frames_sep = [], [] #List of frames that are unique and will be processed
    cat = frame_cat[0]
    for frame in range(1, len(frame_list)):
        if frame_cat[frame] == cat:
            frames_indv.append(frame)
        else:
            frames_sep.append(frames_indv)
            cat = frame_cat[frame]
            frames_indv = [frame]

    frames_keep = []
    for frames in frames_sep:
        rnm_frame = random.choice(frames)
        frames_keep.append(rnm_frame)
    chain_sort = chain.slice(frames_keep, copy=False)
    chain_sort.save_pdb(f'{chain_pdb}/{current_sect}_conformers_unique.pdb')
    print(f'{len(frames_keep)} Unique Structures Found for {current_sect}')

# Declare arguments
parser = argparse.ArgumentParser(description = 'Sort through chain poses to determine potential complexes')
parser.add_argument('-chain', required=False, default='./', help= 'Folder containing PDBs for Chain Conformations')
parser.add_argument('-cores', required=False, default=1, type=int, help='# of cores to use')
parser.add_argument('-length', required=False, default=4, type=int, help='Length of acyl chain to build')

#Import Arguments
args = parser.parse_args()
chain_pdb = args.chain
cores = args.cores
length = args.length

#List all present sections of interest
sect_interest = ['AB', 'BC', 'CD', 'DE', 'EF','FG']   
if length >= 6:
    sect_interest.append('GH')
    if length >= 8:
        sect_interest.append('HI')
    elif length >= 10:
        sect_interest.append('IJ')
elif length != 4:
    raise Exception(f'Length of {length} is not supported')

for n in sect_interest:
    conformer_sort(chain_pdb, n)