from models import encoder
from models import losses
from models import resnet
from models import ssl1

REGISTERED_MODELS = {
    'sim-clr': ssl1.SimCLR,
    'eval': ssl1.SSLEval,
    'semi-supervised-eval': ssl1.SemiSupervisedEval,
}
