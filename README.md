# deepspeaker-pytorch

- pip install -r requirement.txt

## *** Training

- copy the data to your directory
- change the line to point to your train directory

```
train_dir = DeepSpeakerDataset(path = YOUR_PATH_TO_TRAIN_DATA,n_triplets=args.n_triplets,loader = file_loader,transform=transform)
```
- open the terminal and enter ``python3 train_triplet.py --epochs 1`` for example to train one epoch.

## *** Enrollment and Test

- copy the test data to your directory
- edit the path in main funcion of files.py to point to test data, and ``python3 files.py`` to generate file test.txt for each person
- go to each person's directory and create ``enrollment.txt``, manually copy the first 5 lines of ``test.txt`` to ``enrollment.txt`` and delete them from ``test.txt``
- edit ``_path`` in ``enrollment_test.py`` to point to test data
- ``python3 enrollment_test.py --resume [PATH_TO_PTH_MODEL_FILE]``