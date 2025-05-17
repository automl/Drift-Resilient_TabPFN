from ..base_trainer import BaseTrainer


class FT(BaseTrainer):
    """
    Fine-tuning
    """

    def __init__(self, args):
        super().__init__(args)

    def __str__(self):
        return f"FT-K={self.K}-{self.base_trainer_str}"
