#data
DATAPATH = 'data/fed'
MOSAIC_KD_DATA="/path/to/data"
N_PARTIES = 20  #TODO 20
INIT_EPOCHS = 500
LOCAL_CKPTPATH = 'ckpt/fed' #TODO
#local training
LR = 0.0025
LR_MIN = 0.001
BATCHSIZE = 256
NUM_WORKERS = 6

#fed
ROUNDS = 200
LOCAL_PERCENT = 1
OPTIMIZER = 'SGD'
FED_BATCHSIZE = 64
STU_LR = 5e-3 # 5e-3
STU_LR_MIN = 1e-5 # 1e-4
STU_WD = 3e-5
CKPTPATH = './ckpt'
DIS_LR = 1e-4 # 1e-3
GEN_LR = 1e-4 # 1e-3

#generator
GEN_Z_DIM = 100

#print
PRINT_FREQ = 1