import chainer
import chainer.functions as F
import chainer.links as L


class VGG(chainer.Chain):
    def __init__(self, class_labels=1000, train_more_layer=False):
        super(VGG, self).__init__()
        self.train_more_layer = train_more_layer

        with self.init_scope():
            self.base = L.VGG16Layers()
            if self.train_more_layer:
                self.fc6 = L.Linear(512 * 7 * 7, 4096)
                self.fc7 = L.Linear(4096, 4096)
            self.fc8 = L.Linear(4096, class_labels)

    def __call__(self, x, t):
        h = self.predict(x)
        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss

    def predict(self, x):
        if self.train_more_layer:
            h = self.base.extract(x)['pool5']
            h = F.dropout(F.relu(self.fc6(h)))
            h = F.dropout(F.relu(self.fc7(h)))
        else:
            h = self.base.extract(x)['fc7']
        return self.fc8(h)
