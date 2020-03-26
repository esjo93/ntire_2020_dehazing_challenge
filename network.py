import torch
from torch import nn
from torch.nn import functional as F

class _aspp_module(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation, activation=nn.ReLU(inplace=True)):
        super(_aspp_module, self).__init__()
        self.atrous_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.activation(x)


class aspp(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=[1, 6, 12, 18], activation=nn.ReLU(inplace=True)):
        super(aspp, self).__init__()
        self.activation = activation
        self.aspp1 =_aspp_module(in_channels, out_channels, 1, padding=0, dilation=dilation[0], activation=activation)
        self.aspp2 = nn.Sequential(
            nn.ReflectionPad2d(dilation[1]),
            _aspp_module(in_channels, out_channels, 3, padding=0, dilation=dilation[1], activation=activation)
        )
        self.aspp3 = nn.Sequential(
            nn.ReflectionPad2d(dilation[2]),
            _aspp_module(in_channels, out_channels, 3, padding=0, dilation=dilation[2], activation=activation)
        )
        self.aspp4 = nn.Sequential(
            nn.ReflectionPad2d(dilation[3]),
            _aspp_module(in_channels, out_channels, 3, padding=0, dilation=dilation[3], activation=activation)
        )

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=True),
                                             nn.BatchNorm2d(out_channels),
                                             activation)
        self.conv1 = nn.Conv2d(out_channels*5, out_channels, 1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        return self.dropout(x)


class make_dense(nn.Module):
    def __init__(self, in_channels, growth_rate=16, stride=1, dilation=1, batch_norm=False, activation=nn.ReLU(inplace=True)):
        super(make_dense, self).__init__()
        self.conv = _make_conv_layer(in_channels, growth_rate, stride, dilation, batch_norm)
        self.act = activation

    def forward(self, x):
        out = self.conv(x)
        out = self.act(out)
        out = torch.cat((x, out), dim=1)
        return out


class res_block(nn.Module):
    def __init__(self, channels, stride=1, dilations=[2, 2, 1, 1], growth_rate=16, activation=nn.ReLU(inplace=True), batch_norm=False):
        super(res_block, self).__init__()

        modules = []
        input_channels = channels

        for i in range(4):
            modules.append(make_dense(input_channels, growth_rate, stride=stride, dilation=dilations[i], batch_norm=batch_norm, activation=activation))
            input_channels += growth_rate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(input_channels, channels, kernel_size=1, stride=1, padding=0)
        self.act = activation

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out += x
        return out


class encoder(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilations=[2, 2, 1, 1], growth_rate=16, activation=nn.ReLU(inplace=True), batch_norm=True):
        super(encoder, self).__init__()
        self.res_block = res_block(in_channels, stride=1, dilations=dilations, growth_rate=growth_rate, activation=activation, batch_norm=batch_norm)
        self.downsampler = _make_conv_layer(in_channels, out_channels, stride=2, dilation=1, batch_norm=batch_norm)
        self.act = activation
    
    def forward(self, x):
        out = self.res_block(x)
        out = self.downsampler(out)
        out = self.act(out)
        return out

        
class generator(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=[2, 2, 1, 1], activation=nn.ReLU(inplace=True)):
        super(generator, self).__init__()

        # Initial conv block
        gen_encoder = [ nn.ReflectionPad2d(1),
                        nn.Conv2d(in_channels, 32, 3),
                        nn.BatchNorm2d(32),
                        activation,
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(32, 32, 3),
                        nn.BatchNorm2d(32),
                        activation ]
        
        # Downsamplings
        in_ch = 32
        out_ch = 64
        
        for _ in range(4):
            gen_encoder += [encoder(in_ch, out_ch, dilations=dilations, activation=activation)]
            in_ch = out_ch
            out_ch *= 2

        gen_encoder += [res_block(in_ch, dilations=dilations, activation=activation, batch_norm=True), aspp(in_ch, in_ch, activation=activation)]
        self.generator_encoder = nn.Sequential(*gen_encoder)

        # AL(Atmospheric Light)/Resist decoder
        decoder = []
        out_ch = in_ch // 2
        for _ in range(4):
            decoder += [ nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                                activation ]
            in_ch = out_ch
            out_ch //= 2

        decoder += [nn.ReflectionPad2d(1),
                    nn.Conv2d(in_ch, out_channels*2, 3), 
                    activation,
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(out_channels*2, out_channels*2, 3) ]
        
        self.al_resist_decoder = nn.Sequential(*decoder)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()                
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        bottleneck = self.generator_encoder(x)
        al_resist = self.al_resist_decoder(bottleneck)
        atmospheric_light, resistance = F.sigmoid(al_resist[:, :3, :, :]), F.relu(al_resist[:, 3:, :, :], inplace=True)
        
        clear_image = (x - atmospheric_light) * resistance + atmospheric_light

        return clear_image, resistance, atmospheric_light


def _make_conv_layer(in_channels, out_channels, stride, dilation, batch_norm):
        conv_layer = [
            nn.ReflectionPad2d(dilation),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=0, dilation=dilation)
        ]

        if batch_norm:
            conv_layer.append(nn.BatchNorm2d(out_channels))
        
        return nn.Sequential(*conv_layer)