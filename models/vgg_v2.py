import chainer
import chainer.functions as F
import chainer.links as L
from chainer.serializers import npz


class VGG(chainer.Chain):

    def __init__(self, class_labels=100, pretrained_model=False):
        super(VGG, self).__init__()
        with self.init_scope():
            self.base = BaseVGG()
            self.fc6 = L.Linear(None, 512)
            self.fc7 = L.Linear(None, 512)
            self.fc8 = L.Linear(None, class_labels)
            self.bn = L.BatchNormalization(512)

        if pretrained_model:
            npz.load_npz(pretrained_model, self.base)

    def __call__(self, x, t):
        h = self.predict(x)
        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss

    def predict(self, x):
        h = self.base(x)
        h = F.dropout(F.relu(self.fc6(h)))
        h = self.bn(h)
        h = F.dropout(F.relu(self.fc7(h)))
        return self.fc8(h)


class BaseVGG(chainer.Chain):

    def __init__(self):
        super(BaseVGG, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(None, 64, 3, 1, 1)
            self.conv1_2 = L.Convolution2D(64, 64, 3, 1, 1)
            self.conv2_1 = L.Convolution2D(64, 128, 3, 1, 1)
            self.conv2_2 = L.Convolution2D(128, 128, 3, 1, 1)
            self.conv3_1 = L.Convolution2D(128, 256, 3, 1, 1)
            self.conv3_2 = L.Convolution2D(256, 256, 3, 1, 1)
            self.conv3_3 = L.Convolution2D(256, 256, 3, 1, 1)
            self.conv4_1 = L.Convolution2D(256, 512, 3, 1, 1)
            self.conv4_2 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv4_3 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv5_1 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv5_2 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv5_3 = L.Convolution2D(512, 512, 3, 1, 1)

            self.bn1_1 = L.BatchNormalization(64)
            self.bn1_2 = L.BatchNormalization(64)
            self.bn2_1 = L.BatchNormalization(128)
            self.bn2_2 = L.BatchNormalization(128)
            self.bn3_1 = L.BatchNormalization(256)
            self.bn3_2 = L.BatchNormalization(256)
            self.bn3_3 = L.BatchNormalization(256)
            self.bn4_1 = L.BatchNormalization(512)
            self.bn4_2 = L.BatchNormalization(512)
            self.bn4_3 = L.BatchNormalization(512)
            self.bn5_1 = L.BatchNormalization(512)
            self.bn5_2 = L.BatchNormalization(512)
            self.bn5_3 = L.BatchNormalization(512)

    def __call__(self, x):
        h = F.relu(self.bn1_1(self.conv1_1(x)))
        h = F.dropout(h, ratio=0.3)
        h = F.relu(self.bn1_2(self.conv1_2(h)))
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.relu(self.bn2_1(self.conv2_1(h)))
        h = F.dropout(h, ratio=0.4)
        h = F.relu(self.bn2_2(self.conv2_2(h)))
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.relu(self.bn3_1(self.conv3_1(h)))
        h = F.dropout(h, ratio=0.4)
        h = F.relu(self.bn3_2(self.conv3_2(h)))
        h = F.dropout(h, ratio=0.4)
        h = F.relu(self.bn3_3(self.conv3_3(h)))
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.relu(self.bn4_1(self.conv4_1(h)))
        h = F.dropout(h, ratio=0.4)
        h = F.relu(self.bn4_2(self.conv4_2(h)))
        h = F.dropout(h, ratio=0.4)
        h = F.relu(self.bn4_3(self.conv4_3(h)))
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.relu(self.bn5_1(self.conv5_1(h)))
        h = F.dropout(h, ratio=0.4)
        h = F.relu(self.bn5_2(self.conv5_2(h)))
        h = F.dropout(h, ratio=0.4)
        h = F.relu(self.bn5_3(self.conv5_3(h)))
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        return h
