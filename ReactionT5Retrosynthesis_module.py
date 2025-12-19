from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class ReactionT5Retrosynthesis:
    def __init__(self, smiles: str, num_beams: int, num_return_sequences: int):
        self.smiles = smiles
        self.num_beams = num_beams
        self.num_return_sequences = num_return_sequences
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("sagawa/ReactionT5v2-retrosynthesis-USPTO_50k")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("sagawa/ReactionT5v2-retrosynthesis-USPTO_50k").to(self.device)

    def config(self):
        self.inp = self.tokenizer(self.smiles, return_tensors="pt").to(self.device)

    def generation(self):
        output = self.model.generate(**self.inp, num_beams=self.num_beams, num_return_sequences=self.num_return_sequences)
        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True).replace(" ", "").rstrip(".")

        return decoded


def retrosynthesis(smiles: str, num_beams: int = 1, num_return_sequences: int = 1) -> list[str]:    
    retro = ReactionT5Retrosynthesis(smiles, num_beams, num_return_sequences)
    retro.config()
    precursors = retro.generation()
    
    return precursors