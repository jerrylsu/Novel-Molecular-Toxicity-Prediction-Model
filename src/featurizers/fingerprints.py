from typing import Optional, List, Tuple
import os
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
        self.error_num = 0
        if not os.path.exists(self.args.smiles_vocab):
            print(f"The smiles vocab is not exist, and create it ...")
            self.vocab_freq = self._update_vocab_frequency(self.args.smiles_file,
                                                           smiles_vocab=self.args.smiles_vocab,
                                                           sep=",")
        else:
            print(f"Load smiles vocab from smiles_vocab_190w.pt cache.")
            self.vocab_freq = torch.load(self.args.smiles_vocab)
            if self.args.update_smiles_file:
                self.vocab_freq = dict(self.vocab_freq)
                self.vocab_freq = self._update_vocab_frequency(self.args.update_smiles_file,
                                                               smiles_vocab=self.args.update_smiles_vocab,
                                                               sep="\t")
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
                molecule = molecule.split(sep)[0].strip() + f" {str(line_num)}"

                # molecule: <csfpy.Molecule named '0' (id 4294967295, 34 atoms) [0x0000013cd4fa55c0]>
                try:
                    molecule = csfpy.Molecule(molecule)
                except RuntimeError as re:
                    self.error_num += 1
                    continue

                # fingerprint: <csfpy.SparseIntVec of size 114 at [0x00000231bbbd0da0]>
                fingerprint = csfpy.csfp(molecule, 2, 5)

                # len(fingerprint): 114
                # SparsIntVec objects can be converted to lists of integers
                fingerprint_list = fingerprint.toList()

                yield fingerprint_list

    def _update_vocab_frequency(self, smiles_file, smiles_vocab, sep=","):
        fingerprints_generator = self._fingerprints_generator(smiles_file, sep=sep)
        for fp_list in tqdm(fingerprints_generator, desc="FingerPrint List"):
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
        miss_all, total_all, one_hots = 0, 0, []
        with open("../../data/jerry_test.csv") as fp:
            lines = fp.readlines()
            for line in lines:
                line = list(filter(lambda x: x != '', line.split(",")))[2:-1]
                line = list(map(lambda x: int(x), line))
                miss, total, one_hot = self._to_onehot(line)
                miss_all = miss_all + miss
                total_all = total_all + total
        #for fp_list in tqdm(self._fingerprints_generator(self.args.smiles_file), desc="Convert to onehot:"):
        #    one_hots.append(self._to_onehot(fp_list))
        res = miss_all / total_all
        res = list(map(lambda x: sum(x), one_hots))
        # 删除全0

        # 删除重复

        return one_hots


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--smiles_file", type=str, default="../../data/test.csv", help="Path of the smiles file.")
    parser.add_argument("--update_smiles_file", type=str, default="./add_train_dev.smi", help="Path of the smiles file.")
    parser.add_argument("--smiles_vocab", type=str, default="../../data/vocab/smiles_vocab_190w.pt", help="Path of the smiles vocab file.")
    parser.add_argument("--update_smiles_vocab", type=str, default="../../data/vocab/update_smiles_vocab.pt", help="Path of the smiles vocab file.")
    parser.add_argument("--upper", type=int, default=2000000, help="the upper of squeeze vocab.")
    parser.add_argument("--lower", type=int, default=0, help="the lower of squeeze vocab.")
    args = parser.parse_args()
    with open(args.update_smiles_file, 'r') as fp:
        cnt = 0
        while True:
            cnt += 1
            line = fp.readline()
            #if "[P" in line:
            #    print(line)
            line = line.split("\t")[0]
            if not line:
                break
            pass

    mols = csfpy.Molecule.from_file("./Jiang1823Validate.smi")
    print(len(mols))
    for mol in mols:
        #print("Name: " + mol.name + " and ID: " + str(mol.id))
        pass
    fp = FingerPrints(args=args)
    # one_hots = fp.to_onehot()

    pass
