import chainer
import chainer.links as L


class VGG(chainer.Chain):
    def __init__(self, class_labels=1000, freeze_params=True):
        super(VGG, self).__init__()

        with self.init_scope():
            self.base = L.VGG16Layers()
            self.fc8 = L.Linear(4096, class_labels)

        if freeze_params:
            self.base.conv1_1.disable_update()
            self.base.conv1_2.disable_update()
            self.base.conv2_1.disable_update()
            self.base.conv2_2.disable_update()
            self.base.conv3_1.disable_update()
            self.base.conv3_2.disable_update()
            self.base.conv3_3.disable_update()
            self.base.conv4_1.disable_update()
            self.base.conv4_2.disable_update()
            self.base.conv4_3.disable_update()
            self.base.conv5_1.disable_update()
            self.base.conv5_2.disable_update()
            self.base.conv5_3.disable_update()
            self.base.fc6.disable_update()
            self.base.fc7.disable_update()

    def __call__(self, x):
        # h = self.base(x, layers=['fc7'])['fc7']
        h = self.base.extract(x)['fc7']
        return self.fc8(h)
