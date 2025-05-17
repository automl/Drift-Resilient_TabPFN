import argparse
from functools import partial

from .methods.agem.agem import AGEM
from .methods.coral.coral import DeepCORAL
from .methods.erm.erm import ERM
from .methods.ewc.ewc import EWC
from .methods.ft.ft import FT
from .methods.groupdro.groupdro import GroupDRO
from .methods.irm.irm import IRM
from .methods.si.si import SI
from .methods.swa.swa import SWA

from .methods.utils import fix_seeds

print = partial(print, flush=True)


def get_model(**kwargs):
    args = argparse.Namespace(**kwargs)

    fix_seeds(args.random_seed)

    if args.method in ["groupdro", "irm"]:
        args.reduction = "none"
    else:
        args.reduction = "mean"

    method_dict = {
        "groupdro": "GroupDRO",
        "coral": "DeepCORAL",
        "irm": "IRM",
        "ft": "FT",
        "erm": "ERM",
        "ewc": "EWC",
        "agem": "AGEM",
        "si": "SI",
        "swa": "SWA",
    }
    trainer = globals()[method_dict[args.method]](args)
    return trainer
