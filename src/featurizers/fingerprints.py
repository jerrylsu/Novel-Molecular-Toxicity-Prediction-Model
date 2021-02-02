from typing import Optional, List, Tuple
import os
import json
import sys
from argparse import ArgumentParser
from tqdm import tqdm
import torch
try:
    import csfpy
except ModuleNotFoundError as mnfe:
    print(f"Not found the module csfpy: {mnfe.name}")
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))


class FingerPrints(object):
    """
    instance size:   1936962 (≈190W)
    vocabulary size: 220682  (≈22w)
    Most Freq: (1763725584049666355, 1796032)
    """
    def __init__(self, args):
        self.args = args
        self.vocab_freq = {}
        self.molecule_names = []
        self.error_num = 0
        if not os.path.exists(self.args.labels_vocab):
            print(f"The labels vocab is not exist, and create it ...")
            self.labels_dict = self._create_labels_dictinary(self.args.labels_file)
        else:
            print(f"Load labels vocab from labels_vocab.pt cache.")
            self.labels_dict = torch.load(self.args.labels_vocab)
        if not os.path.exists(self.args.smiles_vocab):
            print(f"The smiles vocab is not exist, and create it ...")
            self.vocab_freq = self._update_vocab_frequency(self.args.smiles_file,
                                                           smiles_vocab=self.args.smiles_vocab,
                                                           sep=",")
        else:
            if self.args.update_smiles_file:
                self.vocab_freq = torch.load(self.args.smiles_vocab)
                self.vocab_freq = dict(self.vocab_freq)
                self.vocab_freq = self._update_vocab_frequency(self.args.update_smiles_file,
                                                               smiles_vocab=self.args.update_smiles_vocab,
                                                               sep="\t")
            else:
                print(f"Load smiles vocab from update_smiles_vocab.pt cache.")
                self.vocab_freq = torch.load(self.args.update_smiles_vocab)
        print(f"The size of vocab_freq: {len(self.vocab_freq)}")
        self.vocab_freq_squeezed = self._squeeze_vocab_frequency(self.vocab_freq)
        print(f"The size of vocab_freq_squeezed: {len(self.vocab_freq_squeezed)}")
        self.dict = self._create_dictionary(self.vocab_freq_squeezed)
        print(f"The size of dictionary: {len(self.dict)}")
        print(f"Error num: {self.error_num}")
        pass

    def _fingerprints_generator(self, smiles_file, sep=",") -> List[List[int]]:
        line_num = 0  # len(self.vocab_freq)
        with open(smiles_file, "r") as sf:
            while True:
                molecule = sf.readline()
                if not molecule:
                    break
                line_num += 1
                molecule_name, molecule_id = molecule.strip().split(sep)
                molecule = molecule_name + f" {str(line_num)}"

                # molecule: <csfpy.Molecule named '0' (id 4294967295, 34 atoms) [0x0000013cd4fa55c0]>
                try:
                    molecule = csfpy.Molecule(molecule)
                    self.molecule_names.append(molecule_name)
                except RuntimeError as re:
                    self.error_num += 1
                    continue

                # fingerprint: <csfpy.SparseIntVec of size 114 at [0x00000231bbbd0da0]>
                fingerprint = csfpy.csfp(molecule, 2, 5)

                # len(fingerprint): 114
                # SparsIntVec objects can be converted to lists of integers
                fingerprint_list = fingerprint.toList()

                yield molecule_name, self.labels_dict[molecule_id], fingerprint_list

    def _molecule_to_list(self, molecule: str):
        molecule = csfpy.Molecule(molecule)
        fingerprint = csfpy.csfp(molecule, 2, 5)
        fingerprint_list = fingerprint.toList()
        return fingerprint_list

    def _update_vocab_frequency(self, smiles_file, smiles_vocab, sep=","):
        fingerprints_generator = self._fingerprints_generator(smiles_file, sep=sep)
        for _, _, fp_list in tqdm(fingerprints_generator, desc="FingerPrint List"):
            for elem in fp_list:
                self.vocab_freq[elem] = self.vocab_freq.get(elem, 0) + 1
        vocab_freq_ordered = sorted(self.vocab_freq.items(), key=lambda x: x[1], reverse=True)
        torch.save(vocab_freq_ordered, smiles_vocab)
        return vocab_freq_ordered

    def _squeeze_vocab_frequency(self, vocab_freq):
        vocab_freq_squeezed = list(filter(lambda vocab: self.args.lower < vocab[1] < self.args.upper, vocab_freq))
        return vocab_freq_squeezed

    def _create_dictionary(self, vocab_freq_squeezed):
        return {data[0]: index for index, data in enumerate(vocab_freq_squeezed)}

    def _create_labels_dictinary(self, labels_file):
        labels_dict = {}
        with open(labels_file, 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                molecule_id, molecule_toxicity = line.strip().split(",")
                labels_dict[molecule_id] = 0 if molecule_toxicity == "N" else 1
        torch.save(labels_dict, self.args.labels_vocab)
        return labels_dict

    def _show_duplicate_data(self, molecule_names, labels, one_hots):
        print(f"Total size: {len(one_hots)}")
        print(f"Total size that remove duplicate:"
              f"{len(list(map(lambda x: list(x), list(set(list(map(lambda x: tuple(x), one_hots)))))))}")
        one_hots = [tuple(one_hot) for one_hot in one_hots]
        dic = {}
        for molecule_name, label, onehot in zip(molecule_names, labels, one_hots):
            dic[onehot] = dic.get(onehot, []) + [molecule_name]
        duplicate_data = list(filter(lambda x: len(x[1]) > 1, dic.items()))
        for data in duplicate_data:
            print(data[1])

    def _to_onehot(self, fp_list: Optional[List[int]]) -> Tuple[int, int, List[int]]:
        one_hot = [0] * len(self.dict)
        miss, total = 0, len(fp_list)
        for elem in fp_list:
            try:
                one_hot[self.dict[elem]] = 1
            except KeyError as ke:
                miss += 1
        return miss, total, one_hot

    def to_onehot(self):
        miss_all, total_all, molecule_names, labels, one_hots = 0, 0, [], [], []
        for molecule_name, label, fp_list in tqdm(self._fingerprints_generator(self.args.smiles_file, sep="\t"), desc="Convert to onehot:"):
            miss, total, onehot = self._to_onehot(fp_list)
            molecule_names.append(molecule_name)
            labels.append(label)
            one_hots.append(onehot)
            miss_all = miss_all + miss
            total_all = total_all + total
        print(f"Miss rate: {round(miss_all / total_all, 4) * 100}%")

        aaa = self._molecule_to_list('[2H]C(=O)N(C([2H])([2H])[2H])C([2H])([2H])[2H]')
        print(aaa)
        bbb = self._molecule_to_list('CN(C)C=O')
        print(bbb)
        print(aaa==bbb)
        raise  "Jerry"
        self._show_duplicate_data(molecule_names, labels, one_hots)

        return one_hots


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--smiles_file", type=str, default="../../data/Jiang1823Train.smi", help="Path of the smiles file.")
    parser.add_argument("--labels_file", type=str, default="../../data/LabelTrainValidate.csv", help="Path of the labels file.")
    parser.add_argument("--update_smiles_file", type=str, default=None, help="Path of the smiles file.")
    parser.add_argument("--smiles_vocab", type=str, default="../../data/vocab/update_smiles_vocab.pt", help="Path of the smiles vocab file.")
    parser.add_argument("--labels_vocab", type=str, default="../../data/vocab/labels_vocab.pt", help="Path of the labels vocab file.")
    parser.add_argument("--update_smiles_vocab", type=str, default="../../data/vocab/update_smiles_vocab.pt", help="Path of the smiles vocab file.")
    parser.add_argument("--upper", type=int, default=2000000, help="the upper of squeeze vocab.")
    parser.add_argument("--lower", type=int, default=0, help="the lower of squeeze vocab.")
    args = parser.parse_args()
    fp = FingerPrints(args=args)
    one_hots = fp.to_onehot()
    pass
