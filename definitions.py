import os

local_path = os.path.dirname(os.path.abspath(__file__))
embeddings = local_path+'/embeddings'
encoder = local_path+'/weights/embedding_20_word_encoder_upsample.pth.tar'
decoder = local_path+'/weights/embedding_20_word_decoder_upsample.pth.tar'
rnet = local_path+'/weights/netR.pth.tar'
hnet = local_path+'/weights/netH.pth.tar'
