import os 


raw_data_folder = "/scratch/yang_miao/Articulate3D_Baseline/Mask3D_adapted/data/raw/articulate3d/scans_all"
all_scans = list(os.listdir(raw_data_folder))


train_val_scans_folder = "/scratch/yang_miao/Articulate3D_Baseline/Mask3D_adapted/data/raw/articulate3d/scans"
train_val_scans = list(os.listdir(train_val_scans_folder))

test_scans_folder = "/scratch/yang_miao/Articulate3D_Baseline/Mask3D_adapted/data/raw/articulate3d/test"
test_scans = list(os.listdir(test_scans_folder))

val_scans = ['8f82c394d6', '5656608266', '1ae9e5d2a6', 'a980334473', '27dd4da69e', '88627b561e', '8d563fc2cc', 'e91722b5a3', '4422722c49', '40b56bf310', '38d58a7a31', 'e398684d27', 'e9e16b6043', 'b20a261fdf', '0a76e06478', '5eb31827b7', 'bcd2436daf', 'f8062cb7ce', '5654092cc2', '30f4a2b44d', '25f3b7a318', '1204e08f17', '9859de300f', '6464461276', '324d07a5b3', '2e74812d00', '41b00feddb', 'f3d64c30f8', '1366d5ae89', '8a20d62ac0', 'a29cccc784', '55b2bf8036', '1841a0b525', 'e898c76c1f', '61adeff7d5', '5ee7c22ba0', '8890d0a267', '9071e139d9', '4a1a3a7dc5', '5748ce6f01', '419cbe7c11', 'f5401524e5']

# verify all val scans in all scans 
for scan in val_scans:
    assert scan in all_scans, "{} Not in the data!".format(scan)

# get train sets
train_sets = []
for scan in train_val_scans:
    if scan not in val_scans:
        train_sets.append(scan)
        
print(" num of train scans: ", len(train_sets))
print(" num of val scans: ", len(val_scans))
print(" num of test scans: ", len(test_scans))
   
# print("Number of full train sequences: ", len(full_train_seqs))
# output_file = "/scratch/yang_miao/mask3d_mount/Mask3D_adapted/data/raw/articulate3d/train_split_full.txt"
# with open(output_file, "w") as f:
#     for seq in full_train_seqs:
#         f.write(seq + "\n")

# write down sequences 
train_split_file = "/scratch/yang_miao/Articulate3D_Baseline/Mask3D_adapted/datasets/preprocessing/articulate3d_splits/train.txt"
with open(train_split_file, "w") as f:
    for seq in train_sets:
        f.write(seq + "\n")
        
val_split_file =  "/scratch/yang_miao/Articulate3D_Baseline/Mask3D_adapted/datasets/preprocessing/articulate3d_splits/val.txt"
with open(val_split_file, "w") as f:
    for seq in val_scans:
        f.write(seq + "\n")
        
test_split_file =  "/scratch/yang_miao/Articulate3D_Baseline/Mask3D_adapted/datasets/preprocessing/articulate3d_splits/test.txt"
with open(test_split_file, "w") as f:
    for seq in test_scans:
        f.write(seq + "\n")
