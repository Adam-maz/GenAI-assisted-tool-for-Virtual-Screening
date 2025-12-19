import pandas as pd
from rdkit.Chem import Draw
import IPython


#Elegant function for outputs visulization
def visualize_output(df_mols: pd.DataFrame,
                     maxMols: int = 50,
                     molsPerRow: int = 4,
                     subImgSize: tuple = (450, 450) -> IPython.core.display.Image):

    legends = [
        f"Predicted pKi: {pKi:.2f}\n"
        f"{idx}\n"
        f"Yield: {y:.2f}%\n"
        f"QED: {qed:.2f}"
        for pKi, idx, y, qed in zip(
            df_mols['Predicted_pKi'],
            df_mols['name'],
            df_mols['yield'],
            df_mols['qed']
        )
    ]

    return Draw.MolsToGridImage(
        df_mols['mols'],
        maxMols=maxMols,
        molsPerRow=molsPerRow,
        subImgSize=subImgSize,
        legends=legends
    )