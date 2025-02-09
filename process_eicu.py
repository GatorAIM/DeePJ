import numpy as np
import csv
import os
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import random


class PatientInfo:
    def __init__(self, pat_id: str, pad_token: int, max_num_encs: int, max_num_codes: int):
        self.pat_id = pat_id
        self.enc_timestamps = []
        self.enc_intervals = [] # the time interval between this encounter and the previous encounter
        self.enc_ids = []
        self.enc_labels = []
        self.dx_all_ids = [] # the original dx codes here, separated by list
        self.dx_all_ints = [] # padded dx codes here, padded to num_enc and max_num_codes, i.e., num_enc lists with length max_num_codes
        self.dx_all_pad_masks = [] # dx code padding mask here, padded to num_enc and max_num_codes, i.e., num_enc lists with length max_num_codes
        self.px_all_ids = [] # the original px codes here, separated by list
        self.px_all_ints = [] # padded px codes here, padded to num_enc and max_num_codes, i.e., num_enc lists with length max_num_codes
        self.px_all_pad_masks = [] # px code padding mask here, padded to num_enc and max_num_codes, i.e., num_enc lists with length max_num_codes
        self.pat_label = None  # one single label for the patient, that is the label of the last encounter
        self.pad_token = pad_token # the index of the padding token in the code vocabulary
        self.max_num_encs = max_num_encs # the maximum number of encounters per patient, if the number of encounters is less than this, we will pad the encounters
        self.max_num_codes = max_num_codes # the maximum number of codes in each encounter per feature type, if the number of codes is less than this, we will pad the codes
        self.prior_matrix = torch.zeros((max_num_encs * 2 * max_num_codes, max_num_encs * 2 * max_num_codes)) # the prior matrix (max_num_encs * 2 * max_num_codes by max_num_encs * 2 * max_num_codes)
        
    def sort_encs_by_timestamp(self, num_encs: int):
        # Calculate padding only if needed
        if len(self.enc_ids) < num_encs:
            num_pads = num_encs - len(self.enc_ids)

            # Create padding for each list
            pad_timestamps = [self.enc_timestamps[0]] * num_pads
            pad_enc_ids = ["pad_enc_id"] * num_pads
            pad_enc_labels = ["pad_enc_label"] * num_pads
            pad_dx_all_ids = [["pad_dx_id"]] * num_pads
            pad_dx_all_ints = [[self.pad_token] * self.max_num_codes] * num_pads
            pad_dx_all_pad_masks = [[0] * self.max_num_codes] * num_pads
            pad_px_all_ids = [["pad_px_id"]] * num_pads
            pad_px_all_ints = [[self.pad_token] * self.max_num_codes] * num_pads
            pad_px_all_pad_masks = [[0] * self.max_num_codes] * num_pads

            # Prepend padding
            self.enc_timestamps = pad_timestamps + self.enc_timestamps
            self.enc_ids = pad_enc_ids + self.enc_ids
            self.enc_labels = pad_enc_labels + self.enc_labels
            self.dx_all_ids = pad_dx_all_ids + self.dx_all_ids
            self.dx_all_ints = pad_dx_all_ints + self.dx_all_ints
            self.dx_all_pad_masks = pad_dx_all_pad_masks + self.dx_all_pad_masks
            self.px_all_ids = pad_px_all_ids + self.px_all_ids
            self.px_all_ints = pad_px_all_ints + self.px_all_ints
            self.px_all_pad_masks = pad_px_all_pad_masks + self.px_all_pad_masks

        # Sort only if timestamps are not sorted
        if not self.is_sorted(self.enc_timestamps):
            indices = np.argsort(self.enc_timestamps)[-num_encs:]  # Get indices of sorted timestamps
            self.enc_timestamps = [self.enc_timestamps[i] for i in indices]
            self.enc_ids = [self.enc_ids[i] for i in indices]
            self.enc_labels = [self.enc_labels[i] for i in indices]
            self.dx_all_ids = [self.dx_all_ids[i] for i in indices]
            self.dx_all_ints = [self.dx_all_ints[i] for i in indices]
            self.dx_all_pad_masks = [self.dx_all_pad_masks[i] for i in indices]
            self.px_all_ids = [self.px_all_ids[i] for i in indices]
            self.px_all_ints = [self.px_all_ints[i] for i in indices]
            self.px_all_pad_masks = [self.px_all_pad_masks[i] for i in indices]
        
        # some patient might have more than num_encs encounters, we only take the most recent num_encs encounters
        self.enc_timestamps = self.enc_timestamps[-num_encs:]
        
        # address the edge cases where the encounters are different (exists) but the timestamps are the same
        # we simply plus one minute to all following timestamps
        for i in range(1, len(self.enc_timestamps)):
            if (self.enc_timestamps[i] == self.enc_timestamps[i - 1]) and (self.enc_ids[i-1] != "pad_enc_id"):
                self.enc_timestamps[i] += 1
        
        self.enc_ids = self.enc_ids[-num_encs:]
        self.enc_labels = self.enc_labels[-num_encs:]
        self.dx_all_ids = self.dx_all_ids[-num_encs:]
        self.dx_all_ints = self.dx_all_ints[-num_encs:]
        self.dx_all_pad_masks = self.dx_all_pad_masks[-num_encs:]
        self.px_all_ids = self.px_all_ids[-num_encs:]
        self.px_all_ints = self.px_all_ints[-num_encs:]
        self.px_all_pad_masks = self.px_all_pad_masks[-num_encs:]

        # Update patient label to match the last encounter
        self.pat_label = self.enc_labels[-1]

        # Calculate normalzied intervals from the reference timestamp
        self.enc_intervals = [0] + [self.enc_timestamps[i] - self.enc_timestamps[i - 1] for i in range(1, len(self.enc_timestamps))]

    @staticmethod
    def is_sorted(lst: list) -> bool:
        """Efficiently checks if a list is sorted."""
        return all(x <= y for x, y in zip(lst, lst[1:]))
    
    
class EncounterInfo:
    def __init__(self, pat_id: str, enc_id: str, enc_timestamp: int, expired: bool):
        self.pat_id = pat_id # string, patient id
        self.enc_id = enc_id # string, encounter id
        self.enc_timestamp = enc_timestamp # positive integer, timestamp of the encounter
        self.enc_label = expired # binary, label of the encounter
        self.dx_ids = [] # number of unique dx codes of this encounter
        self.dx_ints = [] # length of max number of dx codes of this encounter, with padding
        self.px_ids = [] # number of unique px codes of this encounter
        self.px_ints = [] # length of max number of dx codes of this encounter, with padding
        self.dx_pad_mask = None
        self.px_pad_mask = None
        
def create_encounters(infile: str, enc_dict: dict, hour_threshold: int = 24) -> dict:
    duration_cut = 0
    dup_cut = 0
    total = 0
    with open(infile, 'r') as f:
        # csv.DictReader treat each row as a dictionary
        for line in csv.DictReader(f):
            total += 1
            pat_id = line['patienthealthsystemstayid']
            enc_id = line['patientunitstayid']
            enc_timestamp = -int(line['hospitaladmitoffset'])
            discharge_status = line['unitdischargestatus']
            expired = 1 if discharge_status=='Expired' else 0
            duration_minute = float(line['unitdischargeoffset'])
            # if the duration is too long, we ignore this encounter
            if duration_minute > 60. * hour_threshold:
                duration_cut += 1
                continue
            # if the encounter already exists, we skip the new one
            if enc_id in enc_dict:
                dup_cut += 1
                continue
            # create the EncounterInfo object to store the encounter information
            enc_dict[enc_id] = EncounterInfo(pat_id, enc_id, enc_timestamp, expired)
    print('Total encounter: %d' % total)
    print('duration_cut: %d' % duration_cut)
    print('dup_cut: %d' % dup_cut)

    return enc_dict

def process_dx(infile: str, enc_dict: dict, source: str) -> dict:
    # update the diagnosis to the EncounterInfo object stored in encounter_dict
    with open(infile, 'r') as f:
        missing_eid = 0
        for line in tqdm(csv.DictReader(f)):
            enc_id = line['patientunitstayid']
            col_name = 'admitdxpath' if source == "admit" else 'diagnosisstring'
            dx_id = line[col_name].lower()
            
            if enc_id not in enc_dict:
                missing_eid += 1
                continue
            enc_dict[enc_id].dx_ids.append(dx_id)
    print('Diagnoses without encounter id: {}'.format(missing_eid))
    return enc_dict

def process_px(infile: str, enc_dict: dict) -> dict:
    # update the treatment to the EncounterInfo object stored in encounter_dict
    with open(infile, 'r') as f:
        missing_eid = 0
        for line in tqdm(csv.DictReader(f)):
            enc_id = line['patientunitstayid']
            px_id = line['treatmentstring'].lower()
            if enc_id not in enc_dict:
                missing_eid += 1
                continue
            enc_dict[enc_id].px_ids.append(px_id)
    print('Treatment without encounter id: {}'.format(missing_eid))
    return enc_dict

def process_encounter_feautures(enc_dict: dict, min_num_codes: int = 1, max_num_codes: int = 50):
    enc_features_list = [] # store all the encounters that satisfy the criteria
    all_str2int = {}  # map each code to an integer
    count = 0
    min_dx_cut = 0
    min_px_cut = 0
    max_dx_cut = 0
    max_px_cut = 0
    num_dx_ids = 0
    num_px_ids = 0
    num_expired = 0
    
    # if the encounter has less than min_num_codes or more than max_num_codes diagnosis or treatment codes, 
    # we skip this encounter    
    for _, enc in enc_dict.items():

        if len(set(enc.dx_ids)) < min_num_codes:
            min_dx_cut += 1
            continue
        if len(set(enc.px_ids)) < min_num_codes:
            min_px_cut += 1
            continue
        if len(set(enc.dx_ids)) > max_num_codes:
            max_dx_cut += 1
            continue
        if len(set(enc.px_ids)) > max_num_codes:
            max_px_cut += 1
            continue
        
        count += 1

        # to map each code to an integer, we create the mapping dict here
        # please note the different feauture types are combined together here
        # generate a bigger vocabulary for all the codes, i.e. all feature type's codes share a vocabulary
        for dx_id in enc.dx_ids:
            if dx_id not in all_str2int:
                all_str2int[dx_id] = len(all_str2int)
        for px_id in enc.px_ids:
            if px_id not in all_str2int:
                all_str2int[px_id] = len(all_str2int)
                
        if enc.enc_label == 1:
            num_expired += 1
            
        # sort the codes and treatments and convert them to integers (we do not assume order within an encounter)
        # please be note that here we cancel the duplicate codes
        dx_ids = sorted(list(set(enc.dx_ids)))
        dx_ints = [all_str2int[item] for item in dx_ids]
        px_ids = sorted(list(set(enc.px_ids)))
        px_ints = [all_str2int[item] for item in px_ids]

        # append/modify the information to the original encounter object
        enc.dx_ids = dx_ids
        enc.dx_ints = dx_ints
        enc.px_ids = px_ids
        enc.px_ints = px_ints
        num_dx_ids += len(enc.dx_ids)
        num_px_ids += len(enc.px_ids)

        enc_features_list.append(enc)
    
    
    # pad the dx and px code to the max num of codes
    pad_token = len(all_str2int) # this is the padding index for diagnosis codes, which equals to vocabulary size 
    
    for enc in enc_features_list:
        enc.dx_ints.extend([pad_token]*(max_num_codes-len(enc.dx_ints)))
        enc.px_ints.extend([pad_token]*(max_num_codes-len(enc.px_ints))) 
        enc.dx_pad_mask = [0 if i==pad_token else 1 for i in enc.dx_ints]
        enc.px_pad_mask = [0 if i==pad_token else 1 for i in enc.px_ints]
        
    print('Number of encounters: %d' % count)
    print('Average number of dx: %f' % (num_dx_ids / count))
    print('Average number of px: %f' % (num_px_ids / count))
    print('Min dx cut: %d' % min_dx_cut)
    print('Min px cut: %d' % min_px_cut)
    print('Max dx cut: %d' % max_dx_cut)
    print('Max px cut: %d' % max_px_cut)
    print('Proportion of expired: %f' % (num_expired / count))

    return enc_features_list, all_str2int

def create_patients(pat_dict: dict, enc_features_list: list, pad_token: int,
                       max_num_encs: int = 2, max_num_codes: int = 50) -> dict:
    
    for enc in enc_features_list:
        pat_id = enc.pat_id
        if pat_id not in pat_dict:
            pat_dict[pat_id] = PatientInfo(pat_id, pad_token, max_num_encs, max_num_codes)
        pat_dict[pat_id].enc_timestamps.append(enc.enc_timestamp)
        pat_dict[pat_id].enc_ids.append(enc.enc_id)
        pat_dict[pat_id].enc_labels.append(enc.enc_label)
        
        pat_dict[pat_id].dx_all_ids.append(enc.dx_ids)
        pat_dict[pat_id].dx_all_ints.append(enc.dx_ints)
        pat_dict[pat_id].dx_all_pad_masks.append(enc.dx_pad_mask)
        
        pat_dict[pat_id].px_all_ids.append(enc.px_ids)
        pat_dict[pat_id].px_all_ints.append(enc.px_ints)
        pat_dict[pat_id].px_all_pad_masks.append(enc.px_pad_mask)
    
    # print some statistics
    num_pats = len(pat_dict)
    
    num_encs = 0
    for pat in pat_dict.values():
        num_encs += len(pat.enc_ids)
    print('Number of patients: %d' % num_pats)
    print('Average number of encounters per patient: %f' % (num_encs / num_pats))
    
    # this process also pad the encounters to the required number of encounters
    for _, pat in (pat_dict.items()):
        pat.sort_encs_by_timestamp(max_num_encs)
        
    # compute the proportion of expired patients
    num_expired = 0
    for pat in pat_dict.values():
        if pat.pat_label == 1:
            num_expired += 1
    print('Proportion of expired: %f' % (num_expired / num_pats))
    
    return pat_dict

def count_single_freq(freqs: dict, ids: list, t:int) -> dict:
    for id in ids:
        key = (t, id)
        if key not in freqs:
            freqs[key] = 0
        freqs[key] += 1
    return freqs

def count_joint_freq(freqs: dict, ids_t1: list, ids_t2: list, t1: int, t2: int) -> dict:
    for id_t1 in ids_t1:
        for id_t2 in ids_t2:
            # rule out the same code in the same time step
            if not (t1 == t2 and id_t1 == id_t2):
                key = ((t1, id_t1), (t2, id_t2))
                if key not in freqs:
                    freqs[key] = 0
                freqs[key] += 1
    return freqs

def compute_conditional_probs(pat_dict: dict, id_set: set):
    # we compute the conditional probability here
    # P(dx_t2 | dx_t1), P(dx_t2 | px_t1), P(px_t2 | dx_t1), P(px_t2 | px_t1)
    # where t1 belongs to [0, max_num_encs], t2 belongs to [t1, max_num_encs]
    # to derive the above conditional probabilities, we need:
    # 1. the frequency of each code at each encounter timestamp, i.e., P(dx_t1), P(px_t1), where t belongs to [0, max_num_encs]
    # 2. the frequency of each pair of codes at each encounter timestamp, i.e., P(dx_t1, dx_t2), P(dx_t1, px_t2), P(px_t1, dx_t2), P(px_t1, px_t2)
    # here we do not consider the time interval between encounters for now (might be added in the future)
    single_freqs = {} # the count of occurrence of each code at each encounter timestamp
    joint_freqs = {}# the count of occurrence of each pair of dx codes at each encounter timestamp pair 
    total_pat = len(pat_dict)
    
    for pat_id, pat in tqdm(pat_dict.items()):
        # only consider the patients in the training set
        if pat_id not in id_set:
            continue
        dx_ids = pat.dx_all_ids
        px_ids = pat.px_all_ids
        max_num_encs = pat.max_num_encs
        
        # please note that here we have the padding token in the dx_ids and px_ids ("pad_dx_id" and "pad_px_id")
        for t in range(max_num_encs):
            dx_ids_t = dx_ids[t]
            px_ids_t = px_ids[t]
            single_freqs = count_single_freq(single_freqs, dx_ids_t, t)
            single_freqs = count_single_freq(single_freqs, px_ids_t, t)

        for t1 in range(max_num_encs):
            for t2 in range(t1, max_num_encs):
                dx_ids_t1 = dx_ids[t1]
                px_ids_t1 = px_ids[t1]
                dx_ids_t2 = dx_ids[t2]
                px_ids_t2 = px_ids[t2]

                joint_freqs = count_joint_freq(joint_freqs, dx_ids_t1, px_ids_t2, t1, t2)
                joint_freqs = count_joint_freq(joint_freqs, px_ids_t1, dx_ids_t2, t1, t2)
                joint_freqs = count_joint_freq(joint_freqs, dx_ids_t1, dx_ids_t2, t1, t2)
                joint_freqs = count_joint_freq(joint_freqs, px_ids_t1, px_ids_t2, t1, t2)
        
    # calculate the unconditional probabilities, i.e. P(D) and P(P) and P(DP)
    single_probs = dict([(k, v / float(total_pat)) for k, v in single_freqs.items()])
    joint_probs = dict([(k, v / float(total_pat)) for k, v in joint_freqs.items()])
    
    # calculate the conditional probabilities
    conditional_probs = {}
    for key in joint_probs.keys():
        ((t1, id_t1), (t2, id_t2)) = key
        # filter out the padding token probability
        if id_t1 == "pad_dx_id" or id_t1 == "pad_px_id" or id_t2 == "pad_dx_id" or id_t2 == "pad_px_id":
            continue
        # address the case where the denominator is zero
        if single_probs[(t1, id_t1)] > 0:
            conditional_probs[key] = joint_probs[key] / single_probs[(t1, id_t1)]
        else:
            conditional_probs[key] = 0
    return conditional_probs

def add_conditional_probs(conditional_probs: dict, pat_dict: dict, id_set: set) -> dict:
    pat_dict_new = {}
    for pat_id, pat in pat_dict.items():
        if pat_id not in id_set:
            continue
        dx_ids = pat.dx_all_ids
        px_ids = pat.px_all_ids
        max_num_codes = pat.max_num_codes
        
        for t1 in range(pat.max_num_encs):
            for t2 in range(t1, pat.max_num_encs):
                for i, dx_id_t1 in enumerate(dx_ids[t1]):
                    for j, dx_id_t2 in enumerate(dx_ids[t2]):
                        key = ((t1, dx_id_t1), (t2, dx_id_t2))
                        pat.prior_matrix[max_num_codes * 2 * t1 + i, max_num_codes * 2 * t2 + j] = 0.0 if key not in conditional_probs else conditional_probs[key] 
                for i, px_id_t1 in enumerate(px_ids[t1]):
                    for j, px_id_t2 in enumerate(px_ids[t2]):
                        key = ((t1, px_id_t1), (t2, px_id_t2))
                        pat.prior_matrix[max_num_codes * (2 * t1 + 1) + i, max_num_codes * (2 * t2 + 1) + j] = 0.0 if key not in conditional_probs else conditional_probs[key]
                for i, dx_id_t1 in enumerate(dx_ids[t1]):
                    for j, px_id_t2 in enumerate(px_ids[t2]):
                        key = ((t1, dx_id_t1), (t2, px_id_t2))
                        pat.prior_matrix[max_num_codes * 2 * t1 + i, max_num_codes * (2 * t2 + 1) + j] = 0.0 if key not in conditional_probs else conditional_probs[key]
                for i, px_id_t1 in enumerate(px_ids[t1]):
                    for j, dx_id_t2 in enumerate(dx_ids[t2]):
                        key = ((t1, px_id_t1), (t2, dx_id_t2))
                        pat.prior_matrix[max_num_codes * (2 * t1 + 1) + i, max_num_codes * 2 * t2 + j] = 0.0 if key not in conditional_probs else conditional_probs[key]
        pat_dict_new[pat_id] = pat
    return pat_dict_new

def flatten_list(l: list) -> list:
    return [item for sublist in l for item in sublist]

def create_causal_mask(max_num_encs: int, max_num_codes: int) -> torch.Tensor:
    # Create a causal mask for the attention mechanism
    causal_mask = np.zeros((max_num_encs * 2 * max_num_codes, max_num_encs * 2 * max_num_codes))
    for t1 in range(max_num_encs):
        for t2 in range(t1, max_num_encs):
            causal_mask[max_num_codes * 2 * t1:max_num_codes * 2 * (t1 + 1), max_num_codes * 2 * t2:max_num_codes * 2 * (t2 + 1)] = 1
    causal_mask = torch.tensor(causal_mask, dtype=torch.long)
    # fill diagonal causal_mask of with 0 becasue we do not attend to a code itself
    causal_mask.fill_diagonal_(0)
    return causal_mask


def convert_features_to_tensors(pat_dict: dict) -> TensorDataset:
    """
    Convert patient feature dictionary into a PyTorch TensorDataset.

    Parameters:
        pat_dict (dict): A dictionary where each value is a patient object
                         containing necessary attributes.

    Returns:
        TensorDataset: A dataset containing tensors for patient data.
    """
    # Map patient IDs (strings) to integers for PyTorch compatibility
    pat_id_mapping = {pat_id: idx for idx, pat_id in enumerate(pat_dict.keys())}
    all_pat_id = torch.tensor([pat_id_mapping[pat_id] for pat_id in pat_dict.keys()], dtype=torch.long)

    # Convert patient labels to a tensor
    all_labels = torch.tensor([pat.pat_label for pat in pat_dict.values()], dtype=torch.long)

    # Convert time intervals, expanding to max_num_codes per encounter
    all_intervals = torch.tensor(
        [[t for t in pat.enc_intervals for _ in range(pat.max_num_codes * 2)] for pat in pat_dict.values()],
        dtype=torch.long
    )

    # Flatten and concatenate diagnostic and procedure codes
    all_code_ints = torch.tensor(
        [flatten_list([pat.dx_all_ints[i] + pat.px_all_ints[i] for i in range(pat.max_num_encs)]) 
         for pat in pat_dict.values()],
        dtype=torch.long
    )

    # Flatten and concatenate padding masks
    all_pad_masks = torch.tensor(
        [flatten_list([pat.dx_all_pad_masks[i] + pat.px_all_pad_masks[i] for i in range(pat.max_num_encs)]) 
         for pat in pat_dict.values()],
        dtype=torch.long
    )

    # Stack prior matrices and transpose
    prior_matrices = torch.stack(
        [pat.prior_matrix.transpose(-2, -1) for pat in pat_dict.values()],
        dim=0
    )

    # Create and transpose causal masks
    all_causal_masks = torch.stack(
        [create_causal_mask(pat.max_num_encs, pat.max_num_codes).transpose(-2, -1) for pat in pat_dict.values()],
        dim=0
    )

    # Create the TensorDataset
    dataset = TensorDataset(all_pat_id, all_code_ints, all_pad_masks, all_causal_masks, all_intervals, prior_matrices, all_labels)

    pat_id_mapping = {v: k for k, v in pat_id_mapping.items()}
    return dataset, pat_id_mapping

def shuffle_pat_dict(pat_dict: dict, rand_seed: int) -> dict:
    # Set the random seed
    random.seed(rand_seed)

    # Shuffle the keys
    pat_ids = list(pat_dict.keys())
    random.shuffle(pat_ids)

    # Recreate the dictionary with shuffled keys
    shuffled_pat_dict = {key: pat_dict[key] for key in pat_ids}
    
    return shuffled_pat_dict

def get_eicu_dataset(data_dir, rand_seed = 888):
    pat_file = os.path.join(data_dir, 'patient.csv')
    admission_dx_file = os.path.join(data_dir, 'admissionDx.csv')
    dx_file = os.path.join(data_dir, 'diagnosis.csv')
    px_file = os.path.join(data_dir, 'treatment.csv')
    
    enc_dict = {}
    enc_dict = create_encounters(pat_file, enc_dict, hour_threshold = 24)
    print("Finish processing encounters")
    enc_dict = process_dx(admission_dx_file, enc_dict, "admit")
    enc_dict = process_dx(dx_file, enc_dict, "in-hospital")
    print("Finish processing diagnoses")
    enc_dict = process_px(px_file, enc_dict)
    print("Finish processing treatments")
    
    min_num_codes = 1
    max_num_encs = 2
    max_num_codes = 50
    
    # process the encounter features into the EncounterInfo object
    enc_features_list, all_str2int = process_encounter_feautures(enc_dict, min_num_codes = min_num_codes, max_num_codes = max_num_codes)
    print("Entire vocabulary size: %d" % len(all_str2int))
    
    pat_dict = dict()
    # here we construct the PatientInfo object, that is we combine the encounters of each patient into one object
    pat_dict = create_patients(pat_dict, enc_features_list, pad_token = len(all_str2int), 
                                  max_num_encs = max_num_encs, max_num_codes = max_num_codes)
    print("Finish constructing PatientInfo")
    # we will divide the train, val and test set latter. For now we use the whole dataset to compute the conditional probabilities
    conditional_probs = compute_conditional_probs(pat_dict, set(pat_dict.keys()))
    # add the conditional probabilities to the PatientInfo object as coordinates and values in the conditional probability matrix
    pat_dict = add_conditional_probs(conditional_probs, pat_dict, set(pat_dict.keys()))
    print("Finish adding conditional probabilities")
    
    # shuffle the patient dict to avoid the time bias
    shuffled_pat_dict = shuffle_pat_dict(pat_dict, rand_seed)
    
    dataset, pat_id_mapping = convert_features_to_tensors(shuffled_pat_dict)
    
    return dataset, enc_dict, shuffled_pat_dict, all_str2int, pat_id_mapping