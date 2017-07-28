import vgg
import mobilenet
import resnet50

net_dict = {
    "vgg": vgg
    "resnet50":resnet50
    "mobilenet":mobilenet
}

def get_basenet(name, inputs):
    net = net_dict[name];
    return net.basenet(inputs);
