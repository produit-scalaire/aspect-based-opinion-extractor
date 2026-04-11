
from typing import Literal



class OpinionExtractor:

    # SET THE FOLLOWING CLASS VARIABLE to "FT" if you implemented a fine-tuning approach
    method: Literal["NOFT", "FT"] = "NOFT"

    # DO NOT MODIFY THE SIGNATURE OF THIS METHOD, add code to implement it
    def __init__(self, cfg) -> None:
        self.cfg = cfg


    # DO NOT MODIFY THE SIGNATURE OF THIS METHOD, add code to implement it
    def train(self, train_data: list[dict], val_data: list[dict]) -> None:
        """
        Trains the model, if OpinionExtractor.method=="FT"
        """
        pass

    # DO NOT MODIFY THE SIGNATURE OF THIS METHOD, add code to implement it
    def predict(self, texts: list[str]) -> list[dict]:
        """
        :param texts: list of reviews from which to extract the opinion values
        :return: a list of dicts, one per input review, containing the opinion values for the 3 aspects.
        """
        pass