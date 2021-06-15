from torch import nn


class PatchDiscriminator(nn.Module):
    def __init__(self, input_c, num_filters=64, n_down=3):
        super().__init__()
        model = [self.get_layers(input_c, num_filters, norm=False)]
        model += [self.get_layers(num_filters * 2 ** i, num_filters * 2 ** (i + 1), s=1 if i == (n_down-1) else 2)
                  for i in range(n_down)]  # the 'if' statement is taking care of not using
        # stride of 2 for the last block in this loop
        # Make sure to not use normalization or
        model += [self.get_layers(num_filters * 2 **
                                  n_down, 1, s=1, norm=False, act=False)]
        # activation for the last layer of the model
        self.model = nn.Sequential(*model)

    # when needing to make some repeatitive blocks of layers,
    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True, act=True):
        # it's always helpful to make a separate method for that purpose
        layers = [nn.Conv2d(ni, nf, k, s, p, bias=not norm)]
        if norm:
            layers += [nn.BatchNorm2d(nf)]
        if act:
            layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
