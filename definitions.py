"""
WARNING - This document and related “CodeNitz” project material may contain technical information, export of which may be restricted by the Export Administration Regulations (EAR).  This project is subject to a license exception, which permits transfer of technical information to certain foreign entities, including IQT’s UK and Australian affiliates, and New Zealand Contractor Josh Bailey and his company OSDL, exclusively for “internal development or production of new products”.  Beyond this, IQT personnel must consult with IQT Legal prior to disclosing any project-related information publicly or to foreign persons.  IQT Legal will continue to review the classification of this technology as it is developed, and will update this marking accordingly.
"""
import os

local_path = os.path.dirname(os.path.abspath(__file__))
embeddings = local_path+'/embeddings'
encoder = local_path+'/weights/embedding_20_word_encoder_upsample.pth.tar'
decoder = local_path+'/weights/embedding_20_word_decoder_upsample.pth.tar'
rnet = local_path+'/weights/netR.pth.tar'
hnet = local_path+'/weights/netH.pth.tar'
