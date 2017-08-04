from vgg16 import Vgg16
from data_gen_helper import create_valid
import os

class Espy():

    def __init__(self, validationSize = 0.2, batchSize = 24):
        self.model = Vgg16()
        self.validationSize = validationSize
        self.batchSize = batchSize

    def fit(self, imageLocation, numEpoch = 1):
        #create validation set
        if os.path.exists(imageLocation + '../valid/') == False:
            valSize = min(self._getValidationSizes(imageLocation))
            create_valid(imageLocation, valSize)

        batches = self.model.get_batches(imageLocation, batch_size = self.batchSize)
        valBatches = self.model.get_batches(imageLocation+'../valid', batch_size = self.batchSize * 2)
        self.model.finetune(batches)
        self.model.fit(batches, valBatches, nb_epoch=numEpoch)

    def predict(self, imageLocation):
        fol = os.listdir(imageLocation)
        imgs = os.listdir(imageLocation + fol[0] + '/')

        batch = self.model.get_batches(imageLocation, batch_size=len(imgs))
        images, labels = next(batch)

        return batch.filenames, self.model.predict(images)

    def _getValidationSizes(self, imageLocation):
        classes = os.listdir(imageLocation)
        imgs = [os.listdir( imageLocation + c ) for c in classes]
        imgs = [len(im) for im in imgs]
        valSize = [round(im * self.validationSize) for im in imgs]
        return valSize
