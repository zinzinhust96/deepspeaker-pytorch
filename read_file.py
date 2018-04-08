import os
from pprint import pprint

path = os.path.dirname(os.path.abspath(__file__)) + '/data/BKRecording'

def get_wav_files():
    hash = {}
    for folder in os.listdir(path):
        hash[folder] = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".wav"):
                    if (folder in os.path.join(root, file)):
                        hash[folder].append(os.path.join(root, file) + '\n')

    with open('files.txt', 'w') as f:
        for person in hash:
            f.writelines(hash[person])
    f.close()

    l = 0
    index_hash = {}
    for person in hash:
        index_hash[person] = []
        for index, item in enumerate(hash[person]):
            # print(index + l, item)
            index_hash[person].append(index + l)
        l = len(hash[person])

    with open('triplet.txt', 'w') as f:
        for person in index_hash:
            for a in index_hash[person]:
                for p in index_hash[person]:
                    if (p != a):
                        for person2 in index_hash:
                            if person2 != person:
                                for n in index_hash[person2]:
                                    f.write('{0} {1} {2}\n'.format(a, p, n))
    f.close()



if __name__ == '__main__':
    get_wav_files()