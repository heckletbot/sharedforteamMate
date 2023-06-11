from torchdrug import data
import pandas as pd
import os
import torch
from torch.utils import data as torch_data

class ScopeSuperFamilyClassify(data.ProteinDataset):
    source_file =  'pdb_scope_db40.csv'
    pdb_dir = 'dbstyle_all-40-2.08'
    #source_file = 'scop40_s.csv'
    #pdb_dir = 'ents'
    processed_file = 'scope_superfamily.pkl.gz'

    label2id_dir = 'pdb_scope_label2id.pkl'
    id2label_dir = 'pdb_scope_id2label.pkl'
    
    splits = ["train", "valid", "test"]
    split_ratio = [0.8, 0.1]
    target_fields = ["superfamily_label"]  # label column

    def __init__(self, path='/home/tangwuguo/datasets/scope40', verbose=1, **kwargs):
        if not os.path.exists(path):
            raise FileExistsError("Unknown path `%s` for SCOPE dataset" % path)
        self.path = path            
        df = pd.read_csv(os.path.join(path, self.source_file))
        pkl_file = os.path.join(path, self.processed_file)
        
        if os.path.exists(pkl_file):  
            # load processed pkl
            self.load_pickle(pkl_file, verbose=verbose, **kwargs)
        else:            
            pdb_files = df['id']
            pdb_files = pdb_files.apply(lambda x: os.path.join(path, self.pdb_dir, x+'.ent')).tolist()
            self.load_pdbs(pdb_files=pdb_files, verbose=1)
            self.save_pickle(pkl_file, verbose=verbose)
                    
        len = df['id'].size
        train_size = int(len*self.split_ratio[0])
        valid_size = int(len*self.split_ratio[1])
        test_size = len - train_size - valid_size
        self.num_samples = [train_size, valid_size, test_size]
        self.targets = {'superfamily_label': torch.tensor(df['label'].tolist())}
    
    
    def split(self, keys=None):
        keys = keys or self.splits
        offset = 0
        splits = []
        for split_name, num_sample in zip(self.splits, self.num_samples):
            if split_name in keys:
                split = torch_data.Subset(self, range(offset, offset + num_sample))
                splits.append(split)
            offset += num_sample
        return splits
        
    
    def get_item(self, index):
        if self.lazy:
            protein = self.load_hdf5(self.pdb_files[index])
        else:
            protein = self.data[index].clone()
        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()
        item = {"graph": protein, "superfamily_label": self.targets["superfamily_label"][index]}
        if self.transform:
            item = self.transform(item)
        return item