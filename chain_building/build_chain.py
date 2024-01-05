import mdtraj as md
import argparse
import math
import os
import numpy as np
from joblib import Parallel, delayed
import time
import subprocess
import glob
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import random

def energy_min(dir, prev_name, i):
    os.chdir(dir)
    if not os.path.exists(f'{prev_name}_{i}'):
        os.mkdir(f'{prev_name}_{i}')
    os.chdir(f'{prev_name}_{i}')
    mdp_file = '../../../min.mdp'
    top_file = 'temp.top'
    gro_file = 'init.gro'
    box_file = 'temp_box.gro'
    tpr_file = 'temp.tpr'
    #Generate topology
    top = subprocess.Popen([gmx_executable, 
                    'pdb2gmx',
                    '-f', f'../{prev_name}_{i}.pdb',
                    '-o', gro_file,
                    '-p', top_file],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE)
    top.communicate(b'1\n1')
    if os.path.exists(gro_file):
        os.remove(f'../{prev_name}_{i}.pdb')
    else:
        top = subprocess.Popen([gmx_executable, 
                    'pdb2gmx',
                    '-f', f'../{prev_name}_{i}.pdb',
                    '-o', gro_file,
                    '-p', top_file],
                    stdin=subprocess.PIPE)
        top.communicate(b'1\n1')
        raise Exception
    #Create a box
    subprocess.run([gmx_executable, 
                    'editconf',
                    '-f', gro_file, 
                    '-o', box_file,
                    '-bt', 'dodecahedron',
                    '-d', '5'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE)
    # Generate TPR
    subprocess.run([gmx_executable,
            'grompp',
            '-f', mdp_file,
            '-p', top_file,
            '-c', box_file,
            '-maxwarn', '1',
            '-o', tpr_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
    if not os.path.exists(tpr_file):
        subprocess.run([gmx_executable,
            'grompp',
            '-f', mdp_file,
            '-p', top_file,
            '-c', box_file,
            '-maxwarn', '1',
            '-o', tpr_file])
        raise Exception
    # Run simulation
    sim = subprocess.call([gmx_executable,
            'mdrun',
            '-deffnm', 'temp',
            '-ntmpi', '1'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
    #Determine if energy minimization ran succesffully
    log = open('temp.log')
    for line in log:
        if 'Steepest Descents' in line:
            if 'machine precision' in line:
                converged = False
            else:
                converged = True
            break

    #Delete pdb if the energy minimization did not converge
    if converged:
        subprocess.run([gmx_executable,
            'genconf',
            '-f', 'temp.gro',
            '-o', f'../{prev_name}_{i}.pdb'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
    os.chdir('../')
    files = glob.glob(f'{prev_name}_{i}/*')
    for f in files:
        os.remove(f)
    os.rmdir(f'{prev_name}_{i}')
    os.chdir('../')
    if not converged:
        return 0
    else:
        return 1
    
def test_confs(align_target, align_section, chain, d, prev_name=None):
    #Determine number of atoms in catalytic triad
    [num_cat_atoms, dim] = np.shape(cat_triad_coors)

    #Set alignment sections for target + chain on serine backbone atoms
    if 'O1' in align_section and d == 0:
        align_atom_indices = align_target.topology.select(f'resid {res_insert-1} and ({align_section.replace("O1", "OG")})')
        chain_atom_indices = chain.topology.select(align_section)
    else:
        align_atom_indices = align_target.topology.select(f'resid {res_insert-1} and ({align_section})')
        chain_atom_indices = chain.topology.select(align_section)
    if len(align_atom_indices) != len(chain_atom_indices):
        raise Exception('Indices in reference do not match chain\n' + str(align_atom_indices) + '\n' + str(chain_atom_indices))
    
    num_pass = 0
    #Determine if any atoms are within 1.5A for all chain structures
    for i in range(chain.n_frames):
        chain_i = chain.slice(i)
    
        #Align chain and target
        chain_i.superpose(align_target, ref_atom_indices=align_atom_indices, atom_indices=chain_atom_indices)
        
        #Get coordinate for this structure
        chain_coordinates = chain_i.xyz[0,:,:]

        #Determine if there is a clash
        clash = False
        for n in range(chain.n_atoms):
            for t in range(atom_potential_clash.n_atoms):
                dist = math.dist(target_coordinates[t,:], chain_coordinates[n,:])
                if dist < 0.15: #Clash present
                    clash = True
                    break
            else:
                continue
            break
        #Determin Threshold for growing away
        if sect_interest[d] == 'AB':
            threshold = 3.5
        elif sect_interest[d] == 'BC':
            threshold = 8
        elif sect_interest[d] == 'CD':
            threshold = 4
        elif sect_interest[d] == 'DE':
            threshold = 3
        else:
            threshold = 2

        #Check that chain isn't growing straight out of the pocket
        grow_away = False
        dist_all = []
        for n in range(chain.n_atoms):
            for t in range(num_cat_atoms):
                dist = math.dist(chain_coordinates[n,:], cat_triad_coors[t,:])
                dist_all.append(dist)
        if all(dist_all) > threshold:
            grow_away = True

        #If no clashed and chain is not growing away
        if clash == False and grow_away == False:
            #save PDB file
            save_pdb(d, i, prev_name, chain_coordinates)

            #Perform energy minimization
            if gmx_executable != None:
                num_pass += energy_min(sect_interest[d], prev_name, i)
            else:
                num_pass += 1
    return num_pass

def get_new_section(file_name, i):
    full_file = open(file_name).readlines()
    for n, line in enumerate(open(file_name)):
        if f'MODEL {str(i).rjust(8, " ")}' in line:
            start=n+1
        if f'MODEL {str(i+1).rjust(8, " ")}' in line:
            end=n-2
            return full_file[start:end]
    return full_file[start:-3]

def write_line(file, line, atom_num, res_name, resid, coord_x, coord_y, coord_z, place1, place2):
    if len(line[2]) > 3:
        add = ' ' 
        x=4
    else:
        add = '  '
        x=3
    
    file.write(line[0] + atom_num.rjust(7, ' ') + add + line[2].ljust(x, ' ') + ' ' + res_name.ljust(3, ' ') + resid.rjust(6, ' ') + coord_x.rjust(12, ' ') + coord_y.rjust(8, ' ') + coord_z.rjust(8, ' ') + place1.rjust(6, ' ') + place2.rjust(6, ' ') + '\n')

def save_pdb(d, i, prev_name, new_coords):
    #Open the previous chain and new chain files
    if d==0:
        prev_file = open(target_pdb)
    else:
        prev_file = open(f'{sect_interest[d-1]}/{prev_name}.pdb')
    
    new_section = get_new_section(f'{chain_pdb}/{sect}_conformers_unique.pdb', i)
    if not os.path.exists(sect_interest[d]):
        os.mkdir(sect_interest[d])
    new_file = open(f'{sect_interest[d]}/{prev_name}_{i}.pdb', 'w')
    new_name = sect_interest[d]
    atoms_already_here = []
    #Loop through lines in file
    added = False
    atom_num = 1
    prev_res = 'NA'
    prev_resid = 'NA'
    for line in prev_file:
        if 'TER' in line:
            new_file.write(line)
        elif 'TITLE' in line or 'REMARK' in line or 'MODEL' in line or 'CRYS' in line or 'ENDMDL' in line:
            continue
        else:
            potential_res_names = ['AB', 'BC', 'CD', 'DE', 'EF', 'FG', 'GH', 'HI', 'C4', 'C10']
            splt_line = []
            for entry in line.split(' '):
                if entry != '':
                    splt_line.append(entry)
            current_res = splt_line[3]
            current_atom = splt_line[2]
            current_resid = splt_line[4]
            if (current_resid == '244' and prev_resid == '243') or (current_resid == '245' and prev_resid == '244' and res_insert == 280): #Seperate FabG and ACP
                new_file.write('TER\n')
            #Just write all lines before we get to the residue of interest
            if added == False and current_res not in potential_res_names and str(res_insert) not in current_resid and prev_res not in potential_res_names and str(res_insert) not in prev_resid:
                new_file.write(line)
                atom_num += 1
            #For atoms in the residue we are changing we need to update to name
            elif added == False and (current_res in potential_res_names or ('SER' in current_res and str(res_insert) in current_resid)) and 'E' not in current_atom and current_atom != 'HG' and current_atom != 'OG':
                write_line(new_file, splt_line, splt_line[1], new_name, splt_line[4], splt_line[5], splt_line[6], splt_line[7], splt_line[8], splt_line[9]) #Change the name for the chain according to the current section added
                atoms_already_here.append(splt_line[2])
                atom_num += 1
            #Add new atoms after previous ones
            elif added == False and (prev_res in potential_res_names or ('SER' in prev_res and str(res_insert) in prev_resid)) and current_res not in potential_res_names and 'SER' not in current_res and str(res_insert) not in current_resid:
                for atom, new_line in enumerate(new_section):
                    splt_new_line = []
                    for entry in new_line.split(' '):
                        if entry != '':
                            splt_new_line.append(entry)
                    new_atom = splt_new_line[2]
                    if 'R' not in new_atom and new_atom not in atoms_already_here and new_atom != 'HB3':
                        write_line(new_file, splt_new_line, str(atom_num), new_name, prev_resid, str(np.round(new_coords[atom,0]*10,3)), str(np.round(new_coords[atom,1]*10,3)), str(np.round(new_coords[atom,2]*10,3)), splt_line[8], splt_line[9])
                        atom_num += 1
                #Add the line we are on after the new atoms
                write_line(new_file, splt_line, str(atom_num), splt_line[3], splt_line[4], splt_line[5], splt_line[6], splt_line[7], splt_line[8], splt_line[9])
                atom_num += 1
                added = True
            #Adjust atom numbering after we add new atoms
            elif added == True:
                write_line(new_file, splt_line, str(atom_num), splt_line[3], splt_line[4], splt_line[5], splt_line[6], splt_line[7], splt_line[8], splt_line[9])
                atom_num += 1
            prev_res = current_res
            prev_resid = current_resid

    if added == False:
        raise Exception('Section not Added to PDB')

def process_file(all_chains, d, n, align_atom):
    #Load target
    if d==0:
        align_target = md.load(n)
    else:
        align_target = md.load(f'{sect_interest[d-1]}/{n}')
    
    total_confs = test_confs(align_target, align_atom, all_chains, d, n.strip('.pdb'))

    num_confs.append(total_confs)

def manual_rmsd(coord_1, coord_2, num_atoms):
    dist = []
    for i in range(num_atoms):
        dist.append(math.dist(coord_1[i], coord_2[i])**2)
    rmsd = 10 * np.sqrt(sum(dist)/4)
    return rmsd

def sort_confs(dir, all_file_names, lc_lim):
    #Load and concatenate pdbs
    traj_all = []
    for i, name in enumerate(all_file_names):
        traj = md.load(dir + '/' + name)
        locals()[f'traj_{i}'] = traj.atom_slice(traj.topology.select('resid > 275 and resid < 281'))
        traj_all.append(locals()[f'traj_{i}'])
    chain = md.join(traj_all)

    #Compute pairwise distances
    distances = np.empty((chain.n_frames, chain.n_frames))
    for i in range(chain.n_frames):
        distances[i] = md.rmsd(chain, chain, i)

    #Perform Clustering
    reduced_distances = squareform(distances, checks=False)
    link = linkage(reduced_distances, method='average') #The hierarchical clustering encoded as a matrix
    frame_list = dendrogram(link, no_labels=False, count_sort='descendent')['leaves']
    frame_cat = dendrogram(link, no_labels=False, count_sort='descendent')['color_list']

    #Keep only one file per cluster
    frames_sep = [] #List of frames that are unique and will be processed
    cat = frame_cat[0]
    frames_indv = [0]
    for frame in range(1, len(frame_list)-1):
        if frame_cat[frame] == cat:
            frames_indv.append(frame)
        else:
            frames_sep.append(frames_indv)
            cat = frame_cat[frame]
            frames_indv = [frame]
    frames_sep.append(frames_indv)

    #Save file names which have unique clusters
    unique_file_names = []
    for i in range(len(frames_sep)):
        if len(frames_sep[i]) > 0:
            num_choice = int(len(frames_sep[i])/lc_lim) #Large clusters select multiple
            if num_choice == 0:
                num_choice = 1
            frames_unique = random.sample(frames_sep[i], num_choice)
            for frame_unique in frames_unique:
                unique_file_names.append(all_file_names[frame_unique])
    for name in all_file_names:
        if name not in unique_file_names:
            os.remove(dir + '/' + name)
    return unique_file_names

# Declare arguments
parser = argparse.ArgumentParser(description = 'Sort through chain poses to determine potential complexes')
parser.add_argument('-t', '-target', required=True, help='PDB for Protein Complex')
parser.add_argument('-c', '-chain', required=False, default='./', help= 'Folder containing PDBs for Chain Conformations')
parser.add_argument('-r', '-cores', required=False, default=1, type=int, help='# of cores to use')
parser.add_argument('-l', '-length', required=False, default=4, type=int, help='Length of acyl chain to build')
parser.add_argument('-s', '-start', required=False, default=0, type=int, help='Length of acyl chain in input file (0 if SER residue only)')
parser.add_argument('-g', '-gmx', required=False, default=None, type=str, help='Executable path for gromacs (Energy minimization will not be performed if not provided)')
parser.add_argument('-i', '-res', required=True, type=int, help='Residue ID to grow chain')


#Import Arguments
args = parser.parse_args()
target_pdb = args.t
chain_pdb = args.c
cores = args.r
length = args.l
start_length = args.s
gmx_executable = args.g
res_insert = args.i

#Load FabG-ACP complex
target = md.load(target_pdb)

#Select atoms indices for which we want to avoid clashes
atom_potential_clash = target.atom_slice(target.topology.select('resid < 277 or resid > 281')) #exclude replaced SER and neighboring residues
target_coordinates = atom_potential_clash.xyz[0,:,:]

#Get coordinates for catalytic triad
cat_triad = target.atom_slice(target.topology.select('resid 136 or resid 149 or resid 153'))
cat_triad_coors = cat_triad.xyz[0]
del target

#Check that length and start_length are compatible
if start_length != 0 and start_length != 4 and start_length != 6 and start_length != 8:
    raise Exception(f'Starting Length of {start_length} is not supported')
if length%2 != 0 or length > 10 or length < 4:
    raise Exception(f'Length of {length} is not supported')

#Determine Sections of Interest
if start_length == 0:
    #List all desired base sections of interest to add
    sect_interest = ['AB', 'BC', 'CD', 'DE', 'EF', 'FG'] #up to C4
    #List alignment atoms for all sections
    align_atoms = ['name N or name CA or name C or name O or name CB', 
                'name O1 or name P or name O2 or name O4', 
                'name C1 or name C2 or name C3',
                'name N1 or name C6 or name O6', 
                'name C7 or name C8 or name C9', 
                'name C10 or name C11 or name S',
                'name C13 or name C14 or name C15',
                'name C15 or name C16 or name C17',
                'name C17 or name C18 or name C19']
else:
    sect_interest = []
    align_atoms = ['name C13 or name C14 or name C15',
                'name C15 or name C16 or name C17',
                'name C17 or name C18 or name C19']
potential_add_sect = ['GH', 'HI', 'IJ']
for s in range(int((length-4)/2)):
    sect_interest.append(potential_add_sect[s])

#Initialize time counter
t = time.time()

#Iteratively add each section or interest
for d, sect in enumerate(sect_interest):
    if os.path.exists(sect_interest[d]) and len(next(os.walk(sect_interest[d]), (None, None, []))[2]) > 0: #Skip section if files already present
        print(f'Skipping {sect_interest[d]}: Using Files Present')

    else:
        #Load target for alignment
        if d == 0:
            align_target_name = [target_pdb]
        elif d < 2:
            align_target_name = next(os.walk(sect_interest[d-1]), (None, None, []))[2]
        else:
            align_target_name_all = next(os.walk(sect_interest[d-1]), (None, None, []))[2]
            #RMSD clustering on growing residue to sort conformers
            align_target_name = sort_confs(sect_interest[d-1], align_target_name_all, lc_lim=6)
            print(f'{len(align_target_name)} Unique Structures Found from Section {sect_interest[d-1]}')

        #Load sections conformations
        sect_conf = md.load(f'{chain_pdb}/{sect}_conformers_unique.pdb')
        
        num_confs = []
        #Parallel(n_jobs=cores)(delayed(process_file)(sect_conf, d, n, align_atoms[d]) for n in align_target_name)
        for n in align_target_name:
            process_file(sect_conf, d, n, align_atoms[d])
        
        print(f'{sect_interest[d]}: {sum(num_confs)} Conformations in {np.round((time.time()-t)/60,2)} minutes')
        t = time.time()
final_name_list = sort_confs(sect_interest[-1], next(os.walk(sect_interest[-1]), (None, None, []))[2], lc_lim=10)

