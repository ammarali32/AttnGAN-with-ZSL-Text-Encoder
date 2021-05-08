# AttnGAN-with-ZSL-Text-Encoder
## Training:
To start training download the data from <a href="https://drive.google.com/file/d/1YUcYHgv4HceHOzza8OGzMp092taKAAq1/view">google Drive</a> and extract them inside the data folder.<br>
<a href="https://github.com/ammarali32/AttnGAN-with-ZSL-Text-Encoder/blob/master/trainer.py">trainer.py </a> has three training functiond the first one will train ZSLGAN, Second one will train DAMSM with ZSL text encoding ,and the third one to train the full Attention GAN.
## Notes:
It is important to add the path of the trained DAMSM model to the <a href="https://github.com/ammarali32/AttnGAN-with-ZSL-Text-Encoder/blob/master/cfg/bird.yml">config file</a> After Stage 2.</br>
The Explanation and further ideas could be found on the <a href="https://github.com/ammarali32/AttnGAN-with-ZSL-Text-Encoder/blob/master/attentionGAN_report.pdf">report</a>
