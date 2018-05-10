import os

# _path = os.path.dirname(os.path.abspath(__file__)) + '/data/test_data'

def generateAllAudioPathForEachPerson(_path):
    for label in os.listdir(_path):
        pathToLabel = _path + '/' + label
        os.chdir(pathToLabel)
        f = open(pathToLabel + '/' + 'test.txt','w')
        for root, dirs, files in os.walk(pathToLabel):
            for file in sorted(files):
                if file.endswith(".wav"):
                    f.write(os.path.join(root,file) + '\n')
        
        f.close()

def readEnrollmentPaths(_path):
    indices = {}
    classes = []
    for label in os.listdir(_path):
        classes.append(label)
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    for label in os.listdir(_path):
        pathToLabel = _path + '/' + label
        with open(pathToLabel + '/' + 'enrollment.txt', 'r') as f:
            indices[class_to_idx[label]] = f.read().splitlines()
    # print(indices)
    return indices, classes

def readTestPaths(_path):
    indices = {}
    classes = []
    for label in os.listdir(_path):
        classes.append(label)
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    for label in os.listdir(_path):
        pathToLabel = _path + '/' + label
        with open(pathToLabel + '/' + 'test.txt', 'r') as f:
            indices[class_to_idx[label]] = f.read().splitlines()
    # print(indices)
    return indices, classes

def create_indices(_path):
    """Returns 2 items: 1 dict contains the arrays of path to the wav files for each person,
    and 1 array of class names
    """
    indices = {}
    classes = []
    for label in os.listdir(_path):
        classes.append(label)
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    for label in os.listdir(_path):
        indices[class_to_idx[label]] = []
        for root, dirs, files in os.walk(_path):
            for file in files:
                if file.endswith(".wav"):
                    if (label in os.path.join(root, file)):
                        indices[class_to_idx[label]].append(os.path.join(root, file))
    return indices, classes

def main():
    generateAllAudioPathForEachPerson(os.path.dirname(os.path.abspath(__file__)) + '/data/test_set')
    print('\n')
    # create_indices('/home/zinzin/Documents/pytorch/deepspeaker-pytorch/data/enrollment_set')

if __name__ == '__main__':
    main()