################ZSL_GAN configs################################
##ref https://github.com/EthanZhu90/ZSL_GAN/blob/master/train_CUB.py
# parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
# parser.add_argument('--splitmode', default='easy', type=str, help='the way to split train/test data: easy/hard')
# parser.add_argument('--manualSeed', type=int, help='manual seed')
# parser.add_argument('--resume',  type=str, help='the model to resume')
# parser.add_argument('--disp_interval', type=int, default=20)
# parser.add_argument('--save_interval', type=int, default=200)
# parser.add_argument('--evl_interval',  type=int, default=40)

device = "cpu"
gpu = '0' 
splitmode = 'easy' 
manualSeed = 11 
resume  = None 
disp_interval = 20
save_interval = 200
evl_interval = 40

""" Training hyper-parameters """
## ref 
# opt.GP_LAMBDA = 10      # Gradient penalty lambda
# opt.CENT_LAMBDA  = 1
# opt.REG_W_LAMBDA = 0.001
# opt.REG_Wz_LAMBDA = 0.0001

# opt.lr = 0.0001
# opt.batchsize = 1000

GP_LAMBDA = 10  
CENT_LAMBDA = 1
REG_W_LAMBDA = 0.001
REG_Wz_LAMBDA = 0.0001
lr = 0.0001
batchsize = 48
"""Testing hyper-parameters """
##ref
# opt.nSample = 60  # number of fake feature for each class
# opt.Knn = 20      # knn: the value of K

nSample = 6  
Knn = 20  

##ref https://github.com/EthanZhu90/ZSL_GAN/blob/master/models.py
# rdc_text_dim = 1000
# z_dim = 100
# h_dim = 4096
rdc_text_dim = 1000
z_dim = 100
h_dim = 4096

VGG_FEATURES_SIZE = 14
IMG_SIZE = 16
INCEPTION_V3_OUTPUT_SIZE = 17


###############AttenGAN configs#################################
##ref https://github.com/taoxugit/AttnGAN/blob/master/code/main.py

cfg_file = "cfg/bird.yml"
data_dir = ""
UPDATE_INTERVAL = 200
checkpoint_model_ZSL = ""





