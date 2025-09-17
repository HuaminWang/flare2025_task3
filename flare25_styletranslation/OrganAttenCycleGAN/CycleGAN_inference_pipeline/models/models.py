from .Organ_attention import Organ_attention
from .Organ_attention_only_cycleGAN import Organ_attention_only_cycleGAN

def create_model(opt):
    print(opt.model)
    if opt.model == 'Organ_attention':
        model = Organ_attention()
    elif opt.model == 'Organ_attention_only_cycleGAN':
        model = Organ_attention_only_cycleGAN()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)

    print(model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
