import os 

partial_train_split_file = "/scratch/yang_miao/mask3d_mount/Mask3D_adapted/data/raw/articulate3d/train_split.txt"

partial_train_seqs = []
with open(partial_train_split_file, "r") as f:
    for line in f:
        partial_train_seqs.append(line.strip())
   
train_seq_dir = "/data/work-gcp-europe-west4-a/Articulate3D_Dataset/scans"
full_seqs = list(os.listdir(train_seq_dir))

seqs_to_add = list(set(full_seqs) - set(partial_train_seqs))
print("Number of sequences to add: ", len(seqs_to_add))
data_source_folder = "/work/yang_miao/Datasets/ScanNetpp/ScanNetpp/data"
destination_folder = "/scratch/yang_miao/mask3d_mount/Mask3D_adapted/data/raw/articulate3d/scans"
# copy the missing files from the source folder to the destination folder
# for seq in seqs_to_add:
#     os.system("cp -r {} {}".format(os.path.join(data_source_folder, seq, "scans/mesh_aligned_0.05.ply"), os.path.join(destination_folder, seq)))
#     print("Copying ", seq)

# generate full train split file
test_split_file = "/scratch/yang_miao/mask3d_mount/Mask3D_adapted/data/raw/articulate3d/test_split.txt"
## load test split
test_seqs = []
with open(test_split_file, "r") as f:
    for line in f:
        test_seqs.append(line.strip())

full_train_seqs = []
for seq in full_seqs:
    if seq not in test_seqs:
        full_train_seqs.append(seq)
        
print("Number of full train sequences: ", len(full_train_seqs))
output_file = "/scratch/yang_miao/mask3d_mount/Mask3D_adapted/data/raw/articulate3d/train_split_full.txt"
with open(output_file, "w") as f:
    for seq in full_train_seqs:
        f.write(seq + "\n")
