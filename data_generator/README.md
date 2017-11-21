# DATA GENERATOR 
1. Convert any images data from directories
into MNIST format automatically.

2. Backup your 
	~/anaconda/lib/python2.7/site-packages/tensorflow/contrib/learn/python/learn/datasets/ 
directory and replace it with new folder in this project with
the same name.

3. Each time when data is updated, please rerun the
	python make_dataset.py [IMAGE WIDTH] [IMAGE HEIGHT]
those two args are what size you want to resize your
raw images to be.

4. If you're unfortunately running without permission
to write to path like /tmp/ships/, please try to change 
the directory specification in make_dataset.py and 
datasets/ship.py to ensure the dirs are writeable.

Happy hacking!
