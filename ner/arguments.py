# -*- coding: utf-8 -*-

"""
Arguments
Created on 2020/5/13 
"""

__author__ = "Yihang Wu"


class Arguments:
    """
    General settings of arguments
    """

    # Device
    device = 'cuda'

    # Path
    raw_data_dir = '../data/raw'
    raw_data_train = raw_data_dir + '/train.csv'
    raw_data_val = raw_data_dir + '/val.csv'
    raw_data_test = raw_data_dir + '/test.csv'

    data_dir = '../data'
    dataset_path = data_dir + '/dataset.pkl'
    lookup_path = data_dir + '/lookup.pkl'
    padded_dataset_path = data_dir + '/padded_dataset.pkl'

    result_dir = '../result'
    event_dir = result_dir + '/event'
    ckpt_dir = result_dir + '/ckpt'
    embed_dir = result_dir + '/embed'

    embed_word_path = embed_dir + '/word.npy'  # 48
    embed_char_path = embed_dir + '/char.npy'  # 48
    embed_pos_path = embed_dir + '/pos.npy'  # 32
    embed_xavier_path = embed_dir + '/xavier.npy'  # 128

    embedding_paths = (  # these embeddings will be concatenated [CHECK IT]
        embed_word_path, embed_char_path, embed_pos_path
    )
    # embedding_paths = (embed_xavier_path,)  # for base comparing

    # Special tokens and corresponding _indexes
    word_pad = ('<pad>', 0)
    word_oov = ('<oov>', 1)
    entity_pad = ('<p>', 0)
    entity_bos = ('<bos>', 1)
    entity_eos = ('<eos>', 2)

    # Train
    use_pretrained_embeddings = True  # [CHECK IT]
    finished_epoch = 0
    num_epochs = 100
    batch_size = 64
    weight_decay = 0.001
    lr = 1e-3
    min_lr = 5e-5
    lr_decay_factor = 0.95

    # Model Common Part
    num_vocabs = None  # set automatically
    num_entities = None  # set automatically
    embed_dim = 128  # embedding size [CHECK IT]
    model_dim = 256

    # Early Stop
    min_delta = 0.
    patience = 6

    # Test
    test_ckpt = ckpt_dir + '/gru_attn_crf_1block/ckpt_epoch_21.pt'  # [CHECK IT]
    test_batch_size = 200
    write_to_csv = True
    csv_dir = result_dir + '/csv'


class AttnCRFArguments(Arguments):
    """
    Arguments for Attention-CRF model
    GammaAttnCRF in model.py
    """
    model_name = 'attn_crf'

    model_dim = 128
    attention_type = 'scaled_dot'
    num_blocks = 1
    num_heads = 4
    ff_hidden_dim = 512
    dropout_rate = 0.2


class GRUCRFArguments(Arguments):
    """
    Arguments for BiGRU-CRF model
    GammaGRUCRF in model.py
    """
    model_name = 'gru_crf'

    gru_hidden_dim = 100


class GRUAttnCRFArguments(Arguments):
    """
    Arguments for BiGRU-Attention-CRF model
    GammaGRUAttnCRF in model.py
    """
    model_name = 'gru_attn_crf'

    attention_type = 'scaled_dot'  # {dot, scaled_dot, cosine, general} tested
    num_blocks = 1  # {1, 2, 3} tested
    num_heads = 4
    ff_hidden_dim = 512
    dropout_rate = 0.2

    gru_hidden_dim = Arguments.model_dim // 2


class GRUArguments(Arguments):
    """
    Arguments for BiGRU model
    GammaGRU in model.py
    """
    model_name = 'gru'

    gru_hidden_dim = 120


class GRUAttnArguments(Arguments):
    """
    Arguments for BiGRU-Attention model
    GammaGRUAttn in model.py
    """
    model_name = 'gru_attn'

    attention_type = 'scaled_dot'

    num_blocks = 1  # {1, 2, 3} tested
    num_heads = 4
    ff_hidden_dim = 512
    dropout_rate = 0.2

    gru_hidden_dim = Arguments.model_dim // 2
