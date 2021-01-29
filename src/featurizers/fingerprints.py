from typing import Optional, List
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
        if not os.path.exists(self.args.smiles_vocab):
            print(f"The smiles vocab is not exist, and create it ...")
            self.fingerprints_generator = self._fingerprints_generator()
            self.vocab_freq = self._vocab_frequency_statistics()
        else:
            print(f"Load smiles vocab from smiles_vocab.pt cache.")
            self.vocab_freq = torch.load(self.args.smiles_vocab)
        print(f"The size of vocab_freq: {len(self.vocab_freq)}")
        self.vocab_freq_squeezed = self._squeeze_vocab_frequency(self.vocab_freq)
        print(f"The size of vocab_freq_squeezed: {len(self.vocab_freq_squeezed)}")
        self.dict = self._create_dictionary(self.vocab_freq_squeezed)
        print(f"The size of dictionary: {len(self.dict)}")
        pass

    def _fingerprints_generator(self) -> List[List[int]]:
        line_num = 0
        with open(self.args.smiles_file, "r") as sf:
            while True:
                molecule = sf.readline()
                if not molecule:
                    break
                molecule = molecule.split(",")[0].strip() + f" {str(line_num)}"
                line_num += 1

                # molecule: <csfpy.Molecule named '0' (id 4294967295, 34 atoms) [0x0000013cd4fa55c0]>
                molecule = csfpy.Molecule(molecule)

                # fingerprint: <csfpy.SparseIntVec of size 114 at [0x00000231bbbd0da0]>
                fingerprint = csfpy.csfp(molecule, 2, 5)

                # len(fingerprint): 114
                # SparsIntVec objects can be converted to lists of integers
                fingerprint = fingerprint.toList()

                yield fingerprint

    def _vocab_frequency_statistics(self):
        vocab_freq = {}
        for fp_list in tqdm(self.fingerprints_generator, desc="FingerPrint List"):
            for elem in fp_list:
                vocab_freq[elem] = vocab_freq.get(elem, 0) + 1
        vocab_freq_ordered = sorted(vocab_freq.items(), key=lambda x: x[1], reverse=True)
        torch.save(vocab_freq_ordered, self.args.smiles_vocab)
        return vocab_freq_ordered

    def _squeeze_vocab_frequency(self, vocab_freq):
        vocab_freq_squeezed = list(filter(lambda vocab: self.args.lower < vocab[1] < self.args.upper, vocab_freq))
        return vocab_freq_squeezed

    def _create_dictionary(self, vocab_freq_squeezed):
        return {data[0]: index for index, data in enumerate(vocab_freq_squeezed)}

    def _to_onehot(self, fp_list: Optional[List[int]]) -> List[int]:
        # molecule = csfpy.Molecule("Cn1cnc2c1c(=O)n(CC(O)CO)c(=O)n2C")
        # fingerprint = csfpy.csfp(molecule, 2, 5)
        # fingerprint_list = fingerprint.toList()
        one_hot = [0] * len(self.vocab)
        for elem in fp_list:
            one_hot[self.vocab[elem]] = 1
        return one_hot


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--smiles_file", type=str, default="./dataset_v1.csv", help="Path of the smiles file.")
    parser.add_argument("--smiles_vocab", type=str, default="../../data/vocab/smiles_vocab.pt", help="Path of the smiles vocab file.")
    parser.add_argument("--upper", type=int, default=20000, help="the upper of squeeze vocab.")
    parser.add_argument("--lower", type=int, default=100, help="the lower of squeeze vocab.")
    args = parser.parse_args()
    fp = FingerPrints(args=args)
    fp._to_onehot()
    pass
