from network.MCBASegFormer.MCBASegFormer import MCBASegFormer

def mcbasegformer(pretrained=True):
    model = MCBASegFormer(pretrained=pretrained)
    return model