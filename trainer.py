from train.train_CUB import train as trainZSL
from train.train_AttnGAN import run_train as trainAttnGAN
from train.train_AttnGAN_complete import train as atttrainer
def trainer():
    #First step is to train ZSL GAN to GET the text encoder
    trainZSL()
    #Replace the nn.embedding used in attention GAN by the trained text encoder and retrain DAMSM
    trainAttnGAN()
    # train Attention GAN with USING ZSL text encoder
    #atttrainer()

trainer()