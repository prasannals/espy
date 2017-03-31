# espy
Simplifying image recognition

** NOTE ** : Currently just for personal use(need to get some copyright stuff sorted out)

## Usage

```
from espy import Espy
es = Espy()
es.fit('C:/data/images/train/')
fileNames, preds = es.predict('C:/data/images/test/')
```

## Requirements

#### Libraries
The following libraries and their respective dependencies should be installed

* Theano (configure .theanorc )
* Keras (configure .keras/keras.json)

#### Folder placement
Images belonging to different categories should be placed in different folders.
The folder name will be the name of the category.

The folder being passed into the "fit" method should contain all of these different category folders(category folders should be direct subfolders of the folder being passed in).

The folder being passed into the "predict" method should contain a single folder within it which in turn should contain all the test set images.

### Acknowledgements
fast.ai website and their deep learning course from which I've learned whatever deep learning that was required to implement this project.
