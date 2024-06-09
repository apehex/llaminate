"""Setup the hyper parameters for llaminate."""

def rates(pretrained: bool=False, base: float=0.001) -> tuple:
    return (
        (0.1 if pretrained else 1.) * 0.001 * base, # lr min
        (0.1 if pretrained else 1.) * 0.1 * base, # lr max
        0.8) # lr decay rate

def version(num_layers: int, hidden_dim: int) -> list:
    return ['x'.join(str(num_layers), str(hidden_dim))]
