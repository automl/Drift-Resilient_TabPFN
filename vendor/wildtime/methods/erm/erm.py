from ..base_trainer import BaseTrainer


class ERM(BaseTrainer):
    """
    Empirical Risk Minimization
    """

    def __init__(self, args):
        super().__init__(args)

    def __str__(self):
        if self.lisa:
            return f"ERM-LISA-no-domainid-{self.base_trainer_str}"
        elif self.mixup:
            return f"ERM-Mixup-no-domainid-{self.base_trainer_str}"
        return f"ERM-{self.base_trainer_str}"
