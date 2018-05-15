DATASET_DIR = './voxceleb'
AUDIO_DIR = './voxceleb/voxceleb1_wav'



NUM_PREVIOUS_FRAME = 20
#NUM_PREVIOUS_FRAME = 13
NUM_NEXT_FRAME = 14

NUM_FRAMES = NUM_PREVIOUS_FRAME + NUM_NEXT_FRAME
USE_LOGSCALE = True
USE_DELTA = False
USE_SCALE = False
SAMPLE_RATE = 16000
TRUNCATE_SOUND_FIRST_SECONDS = 0.5
FILTER_BANK = 40

