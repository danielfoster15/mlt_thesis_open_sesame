# -*- coding: utf-8 -*-
import json
import math
import os
import sys
import time
import dynet as dy
from optparse import OptionParser

from evaluation import *
from discrete_argid_feats import ArgPosition, OutHeads, SpanWidth
from raw_data import make_data_instance
from semafor_evaluation import convert_conll_to_frame_elements

reload(sys)
sys.setdefaultencoding('UTF8')

optpr = OptionParser()
optpr.add_option("--testf", dest="test_conll", help="Annotated CoNLL test file", metavar="FILE", default=TEST_CONLL)
optpr.add_option("--mode", dest="mode", type="choice", choices=["train", "test", "refresh", "ensemble", "predict"],
                 default="train")
optpr.add_option("--saveensemble", action="store_true", default=False)
optpr.add_option("-n", "--model_name", help="Name of model directory to save model to.")
optpr.add_option("--exemplar", action="store_true", default=False)
optpr.add_option("--spanlen", type="choice", choices=["clip", "filter"], default="clip")
optpr.add_option("--loss", type="choice", choices=["log", "softmaxm", "hinge"], default="softmaxm")
optpr.add_option("--cost", type="choice", choices=["hamming", "recall"], default="recall")
optpr.add_option("--roc", type="int", default=2)
optpr.add_option("--hier", action="store_true", default=False)
optpr.add_option("--syn", type="choice", choices=["dep", "constit", "depconstit", "none"], default="none")
optpr.add_option("--ptb", action="store_true", default=False)
optpr.add_option("--raw_input", type="str", metavar="FILE")
optpr.add_option("--config", type="str", metavar="FILE")
optpr.add_option("--character_based", action = "store_true")
optpr.add_option("--no_data_fes", action = "store_true")
optpr.add_option("--dynet-gpu", action="store_true")
(options, args) = optpr.parse_args()

model_dir = "logs/{}/".format(options.model_name)
model_file_name = "{}best-argid-{}-model".format(model_dir, VERSION)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if options.no_data_fes:
    train_conll = TRAIN_FTE_NO_DATA_FES
    options.test_conll = TEST_CONLL_NO_DATA_FES
    frame_dir = FRAME_DIR_NO_DATA_FES
else:
    frame_dir = FRAME_DIR
    train_conll = TRAIN_FTE
    train_constits = TRAIN_FTE_CONSTITS

USE_SPAN_CLIP = (options.spanlen == "clip")
USE_DROPOUT = True
FE_CLASSIFIER = True
if options.mode in ["test", "predict"]:
    USE_DROPOUT = False
USE_WV = True
if options.character_based:
    USE_CHV = True
    print('using character-based model')
else:
    USE_CHV = False
USE_HIER = options.hier
USE_DEPS = USE_CONSTITS = False
if options.syn == "dep" or options.syn == "depconstit":
    USE_DEPS = True
elif options.syn == "constit" or options.syn == "depconstit":
    USE_CONSTITS = True
USE_PTB_CONSTITS = options.ptb
SAVE_FOR_ENSEMBLE = (options.mode == "test") and options.saveensemble
RECALL_ORIENTED_COST = options.roc

sys.stderr.write("_____________________\n")
sys.stderr.write("COMMAND: {}\n".format(" ".join(sys.argv)))
if options.mode in ["train", "refresh"]:
    sys.stderr.write("VALIDATED MODEL SAVED TO:\t{}\n".format(model_file_name))
else:
    sys.stderr.write("MODEL FOR TEST / PREDICTION:\t{}\n".format(model_file_name))
    sys.stderr.write("SAVING ENSEMBLE?\t{}\n".format(SAVE_FOR_ENSEMBLE))
sys.stderr.write("PARSING MODE:\t{}\n".format(options.mode))
sys.stderr.write("_____________________\n\n")

if USE_PTB_CONSTITS:
    ptbexamples = read_ptb()

trainexamples, _, _ = read_conll(train_conll, options.syn)
post_train_lock_dicts()

frmfemap, corefrmfemap, _ = read_frame_maps(frame_dir)

#use word vectors
if USE_WV:
    wvs = get_wvec_map()
    PRETDIM = len(wvs.values()[0])

chvs = get_chvec_map(use_pca=True)
PRETCHDIM = len(chvs.values()[0])
#user hierarchy of frame relations
if USE_HIER:
    frmrelmap, feparents = read_frame_relations()

lock_dicts()


# Default labels - in CoNLL format these correspond to _
UNKTOKEN = VOCDICT.getid(UNK)
UNKCHAR = CHARDICT.getid(UNKCHR)
CHARPAD= CHARDICT.getid(CHRPAD)
CHARSPACE = CHARDICT.getid(CHRSPACE)
NOTANLU = LUDICT.getid(EMPTY_LABEL)
NOTANFEID = FEDICT.getid(EMPTY_FE)  # O in CoNLL format.


if options.mode in ["train", "refresh"]:
    if options.no_data_fes:
        devexamples, _, _ = read_conll(DEV_CONLL_NO_DATA_FES, options.syn)
    else:
        devexamples, _, _ = read_conll(DEV_CONLL, options.syn)
    out_conll_file = "{}predicted-{}-argid-dev.conll".format(model_dir, VERSION)
elif options.mode in ["test", "ensemble"]:
    devexamples, _, _ = read_conll(options.test_conll, options.syn)
    out_conll_file = "{}predicted-{}-argid-test.conll".format(model_dir, VERSION)
    fe_file = "{}predicted-{}-argid-test.fes".format(model_dir, VERSION)
    if SAVE_FOR_ENSEMBLE:
        out_ens_file = "{}ensemble.{}".format(model_dir, out_conll_file.split("/")[-1][:-11])
    if options.mode == "ensemble":
        in_ens_file = "{}full-ensemble-{}".format(model_dir, out_conll_file.split("/")[-1][:-11])
elif options.mode == "predict":
    assert options.raw_input is not None
    instances, _, _ = read_conll(options.raw_input)
    out_conll_file = "{}predicted-args.conll".format(model_dir)
else:
    raise Exception("Invalid parser mode", options.mode)

# Default configurations.
if USE_CHV:
    configuration = {"train": train_conll,
                     "use_exemplar": options.exemplar,
                     "use_hierarchy": USE_HIER,
                     "use_span_clip": USE_SPAN_CLIP,
                     "allowed_max_span_length": 20,
                     "allowed_max_character_span_length": 20,
                     "using_dependency_parses": USE_DEPS,
                     "using_constituency_parses": USE_CONSTITS,
                     "using_scaffold_loss": USE_PTB_CONSTITS,
                     "loss_type": options.loss,
                     "cost_type": options.cost,
                     "recall_oriented_cost": RECALL_ORIENTED_COST,
                     "unk_prob": 0.1,
                     "dropout_rate": 0.01,
                     "token_dim": 60,
                     "character_dim": 50,
                     "pos_dim": 4,
                     "lu_dim": 64,
                     "lu_pos_dim": 2,
                     "frame_dim": 100,
                     "fe_dim": 50,
                     "phrase_dim": 16,
                     "path_lstm_dim": 64,
                     "path_dim": 64,
                     "dependency_relation_dim": 8,
                     "lstm_input_dim": 64,
                     "chlstm_input_dim": 50,
                     "lstm_dim": 64,
                     "chlstm_dim": 50,
                     "lstm_depth": 1,
                     "hidden_dim": 64,
                     "hidden_ch_dim": 50,
                     "use_dropout": USE_DROPOUT,
                     "pretrained_embedding_dim": PRETDIM,
                     "pretrained_character_embeddings_dim": PRETCHDIM,
                     "num_epochs": 15 if not options.exemplar else 25,
                     "patience": 3,
                     "eval_after_every_epochs": 100,
                     "dev_eval_epoch_frequency": 5}
else:
    configuration = {"train": train_conll,
                     "use_exemplar": options.exemplar,
                     "use_hierarchy": USE_HIER,
                     "use_span_clip": USE_SPAN_CLIP,
                     "allowed_max_span_length": 20,
                     "using_dependency_parses": USE_DEPS,
                     "using_constituency_parses": USE_CONSTITS,
                     "using_scaffold_loss": USE_PTB_CONSTITS,
                     "loss_type": options.loss,
                     "cost_type": options.cost,
                     "recall_oriented_cost": RECALL_ORIENTED_COST,
                     "unk_prob": 0.1,
                     "dropout_rate": 0.01,
                     "token_dim": 60,
                     "pos_dim": 4,
                     "lu_dim": 64,
                     "lu_pos_dim": 2,
                     "frame_dim": 100,
                     "fe_dim": 50,
                     "phrase_dim": 16,
                     "path_lstm_dim": 64,
                     "path_dim": 64,
                     "dependency_relation_dim": 8,
                     "lstm_input_dim": 64,
                     "lstm_dim": 64,
                     "lstm_depth": 1,
                     "hidden_dim": 64,
                     "use_dropout": USE_DROPOUT,
                     "pretrained_embedding_dim": PRETDIM,
                     "num_epochs": 10 if not options.exemplar else 25,
                     "patience": 3,
                     "eval_after_every_epochs": 100,
                     "dev_eval_epoch_frequency": 5}
configuration_file = os.path.join(model_dir, "configuration.json")
if options.mode == "train":
    if options.config:
        config_json = open(options.config, "r")
        configuration = json.load(config_json)
    with open(configuration_file, "w") as fout:
        fout.write(json.dumps(configuration))
        fout.close()
else:
    json_file = open(configuration_file, "r")
    configuration = json.load(json_file)


UNK_PROB = configuration["unk_prob"]
DROPOUT_RATE = configuration["dropout_rate"]
ALLOWED_SPANLEN = configuration["allowed_max_span_length"]

TOKDIM = configuration["token_dim"]
POSDIM = configuration["pos_dim"]
LUDIM = configuration["lu_dim"]
LUPOSDIM = configuration["lu_pos_dim"]

FRMDIM = configuration["frame_dim"]
FEDIM = configuration["fe_dim"]
INPDIM = TOKDIM + POSDIM + 1

if USE_CHV:
    ALLOWED_CHAR_SPANLEN = configuration["allowed_max_character_span_length"]
    CHDIM = configuration["character_dim"]
    CHINPDIM = CHDIM + 1


PATHLSTMDIM = configuration["path_lstm_dim"]
PATHDIM = configuration["path_dim"]

if USE_CONSTITS:
    PHRASEDIM = configuration["phrase_dim"]

LSTMINPDIM = configuration["lstm_input_dim"]
LSTMDIM = configuration["lstm_dim"]
LSTMDEPTH = configuration["lstm_depth"]
HIDDENDIM = configuration["hidden_dim"]
if USE_CHV:
    CHLSTMINPDIM = configuration["chlstm_input_dim"]
    CHLSTMDIM = configuration["chlstm_dim"]
    HIDDENCHDIM = configuration["hidden_ch_dim"]

ARGPOSDIM = ArgPosition.size()
SPANDIM = SpanWidth.size()
if USE_CHV:

    ALL_FEATS_DIM = 2 * LSTMDIM \
                    + LUDIM \
                    + LUPOSDIM \
                    + FRMDIM \
                    + CHLSTMINPDIM \
                    + CHLSTMDIM \
                    + FEDIM \
                    + ARGPOSDIM \
                    + SPANDIM \
                    + 2  # spanlen and log spanlen features and is a constitspan

else:
    ALL_FEATS_DIM = 2 * LSTMDIM \
                    + LUDIM \
                    + LUPOSDIM \
                    + FRMDIM \
                    + LSTMINPDIM \
                    + LSTMDIM \
                    + FEDIM \
                    + ARGPOSDIM \
                    + SPANDIM \
                    + 2  # spanlen and log spanlen features and is a constitspan
if USE_DEPS:
    DEPHEADDIM = LSTMINPDIM + POSDIM
    DEPRELDIM = configuration["dependency_relation_dim"]
    OUTHEADDIM = OutHeads.size()

    PATHLSTMINPDIM = DEPHEADDIM + DEPRELDIM
    ALL_FEATS_DIM += OUTHEADDIM + PATHDIM

if USE_CONSTITS:
    ALL_FEATS_DIM += 1 + PHRASEDIM
    ALL_FEATS_DIM += PATHDIM

NUMEPOCHS = configuration["num_epochs"]
PATIENCE = configuration["patience"]
LOSS_EVAL_EPOCH = configuration["eval_after_every_epochs"]
DEV_EVAL_EPOCHS = configuration["dev_eval_epoch_frequency"] * LOSS_EVAL_EPOCH

trainexamples = filter_long_ex(trainexamples, USE_SPAN_CLIP, ALLOWED_SPANLEN, NOTANFEID)

sys.stderr.write("\nPARSER SETTINGS (see {})\n_____________________\n".format(configuration_file))
for key in sorted(configuration):
    sys.stderr.write("{}:\t{}\n".format(key.upper(), configuration[key]))

sys.stderr.write("\n")

def print_data_status(fsp_dict, vocab_str):
    sys.stderr.write("# {} = {}\n\tUnseen in dev/test = {}\n\tUnlearnt in dev/test = {}\n".format(
        vocab_str, fsp_dict.size(), fsp_dict.num_unks()[0], fsp_dict.num_unks()[1]))

print_data_status(VOCDICT, "Tokens")
print_data_status(POSDICT, "POS tags")
print_data_status(LUDICT, "LUs")
print_data_status(LUPOSDICT, "LU POS tags")
print_data_status(FRAMEDICT, "Frames")
print_data_status(FEDICT, "FEs")
print_data_status(CLABELDICT, "Constit Labels")
print_data_status(DEPRELDICT, "Dependency Relations")
sys.stderr.write("\n_____________________\n\n")


if USE_DEPS:
    DEPHEADDIM = LSTMINPDIM + POSDIM
    DEPRELDIM = configuration["dependency_relation_dim"]
    OUTHEADDIM = OutHeads.size()

    PATHLSTMINPDIM = DEPHEADDIM + DEPRELDIM
    ALL_FEATS_DIM += OUTHEADDIM + PATHDIM

if USE_CONSTITS:
    ALL_FEATS_DIM += 1 + PHRASEDIM  # is a constit and what is it
    ALL_FEATS_DIM += PATHDIM

NUMEPOCHS = configuration["num_epochs"]
PATIENCE = configuration["patience"]
LOSS_EVAL_EPOCH = configuration["eval_after_every_epochs"]
DEV_EVAL_EPOCHS = configuration["dev_eval_epoch_frequency"] * LOSS_EVAL_EPOCH

trainexamples = filter_long_ex(trainexamples, USE_SPAN_CLIP, ALLOWED_SPANLEN, NOTANFEID)

sys.stderr.write("\nPARSER SETTINGS (see {})\n_____________________\n".format(configuration_file))
for key in sorted(configuration):
    sys.stderr.write("{}:\t{}\n".format(key.upper(), configuration[key]))

sys.stderr.write("\n")

def print_data_status(fsp_dict, vocab_str):
    sys.stderr.write("# {} = {}\n\tUnseen in dev/test = {}\n\tUnlearnt in dev/test = {}\n".format(
        vocab_str, fsp_dict.size(), fsp_dict.num_unks()[0], fsp_dict.num_unks()[1]))

print_data_status(VOCDICT, "Tokens")
print_data_status(POSDICT, "POS tags")
print_data_status(LUDICT, "LUs")
print_data_status(LUPOSDICT, "LU POS tags")
print_data_status(FRAMEDICT, "Frames")
print_data_status(FEDICT, "FEs")
print_data_status(CLABELDICT, "Constit Labels")
print_data_status(DEPRELDICT, "Dependency Relations")
sys.stderr.write("\n_____________________\n\n")
model = dy.Model()
adam = dy.AdamTrainer(model, 0.0005, 0.01, 0.9999, 1e-8)
#lookup dictionary thing for tokens in the vocabulary
################################################################
v_x = model.add_lookup_parameters((VOCDICT.size(), TOKDIM))
################################################################
#lookup dictionary for characterset
ch_x = model.add_lookup_parameters((VOCDICT.size(), CHDIM))

#lookup dictionary thing for parts of speech
p_x = model.add_lookup_parameters((POSDICT.size(), POSDIM))
#lookup dictionary for LUs based on the LUDICT dictionary of base LU words
lu_x = model.add_lookup_parameters((LUDICT.size(), LUDIM))
#lookup dictionary for LU parts of speech
lp_x = model.add_lookup_parameters((LUPOSDICT.size(), LUPOSDIM))
#lookup dictionary for frames
frm_x = model.add_lookup_parameters((FRAMEDICT.size(), FRMDIM))
frm_bool_x = model.add_lookup_parameters((2, FRMDIM))
#lookup dictionary for BIOS tags
ap_x = model.add_lookup_parameters((ArgPosition.size(), ARGPOSDIM))
#lookup dictionary for how long a sentence is?
sp_x = model.add_lookup_parameters((SpanWidth.size(), SPANDIM))

if USE_DEPS:
    dr_x = model.add_lookup_parameters((DEPRELDICT.size(), DEPRELDIM))
    oh_s = model.add_lookup_parameters((OutHeads.size(), OUTHEADDIM))

if USE_CONSTITS:
    ct_x = model.add_lookup_parameters((CLABELDICT.size(), PHRASEDIM))
#lookup dict for FEs from FEDICT
fe_x = model.add_lookup_parameters((FEDICT.size(), FEDIM))
#use word vectors
################################################################
if USE_WV:
    e_x = model.add_lookup_parameters((VOCDICT.size(), PRETDIM))
    for wordid in wvs:
        #dict of all vectors for each word
        e_x.init_row(wordid, wvs[wordid])
    w_e = model.add_parameters((LSTMINPDIM, PRETDIM))
    b_e = model.add_parameters((LSTMINPDIM, 1))
################################################################
if USE_CHV:
    #dict of all vectors for each character
    ch_x = model.add_lookup_parameters((CHARDICT.size(), PRETCHDIM))
    for chid in chvs:
        ch_x.init_row(chid, chvs[chid])
    ch_e = model.add_parameters((CHLSTMINPDIM, PRETCHDIM))
    bch_e = model.add_parameters((CHLSTMINPDIM, 1))
#
w_i = model.add_parameters((LSTMINPDIM, INPDIM))
b_i = model.add_parameters((LSTMINPDIM, 1))

if USE_CHV:
    ch_i = model.add_parameters((CHLSTMINPDIM, CHINPDIM))
    bch_i = model.add_parameters((CHLSTMINPDIM, 1))
################################################################
builders = [
    dy.LSTMBuilder(LSTMDEPTH, LSTMINPDIM, LSTMDIM, model),
  dy.LSTMBuilder(LSTMDEPTH, LSTMINPDIM, LSTMDIM, model)
]
################################################################
if USE_CHV:
    chbuilders = [
        dy.LSTMBuilder(LSTMDEPTH, CHLSTMINPDIM, CHLSTMDIM, model),
    dy.LSTMBuilder(LSTMDEPTH, CHLSTMINPDIM, CHLSTMDIM, model)
    ]
################################################################
basefwdlstm = dy.LSTMBuilder(LSTMDEPTH, LSTMINPDIM, LSTMINPDIM, model)
baserevlstm = dy.LSTMBuilder(LSTMDEPTH, LSTMINPDIM, LSTMINPDIM, model)
################################################################
if USE_CHV:
    charfwdlstm = dy.LSTMBuilder(LSTMDEPTH, CHLSTMINPDIM, CHLSTMINPDIM, model)
    charrevlstm = dy.LSTMBuilder(LSTMDEPTH, CHLSTMINPDIM, CHLSTMINPDIM, model)
################################################################
w_bi = model.add_parameters((LSTMINPDIM, 2 * LSTMINPDIM))
b_bi = model.add_parameters((LSTMINPDIM, 1))
################################################################
if USE_CHV:
    ch_bi = model.add_parameters((CHLSTMINPDIM, 2 * CHLSTMINPDIM))
    bch_bi = model.add_parameters((CHLSTMINPDIM, 1))


################################################################
tgtlstm = dy.LSTMBuilder(LSTMDEPTH, LSTMINPDIM, LSTMDIM, model)
ctxtlstm = dy.LSTMBuilder(LSTMDEPTH, LSTMINPDIM, LSTMDIM, model)
#################################################################
if USE_CHV:
    chtgtlstm = dy.LSTMBuilder(LSTMDEPTH, CHLSTMINPDIM, CHLSTMDIM, model)
    chctxtlstm = dy.LSTMBuilder(LSTMDEPTH, CHLSTMINPDIM, CHLSTMDIM, model)

if USE_DEPS:
    w_di = model.add_parameters((LSTMINPDIM, LSTMINPDIM + DEPHEADDIM + DEPRELDIM))
    b_di = model.add_parameters((LSTMINPDIM, 1))

    pathfwdlstm = dy.LSTMBuilder(LSTMDEPTH, LSTMINPDIM, PATHLSTMDIM, model)
    pathrevlstm = dy.LSTMBuilder(LSTMDEPTH, LSTMINPDIM, PATHLSTMDIM, model)

    w_p = model.add_parameters((PATHDIM, 2 * PATHLSTMDIM))
    b_p = model.add_parameters((PATHDIM, 1))
elif USE_CONSTITS:
    cpathfwdlstm = dy.LSTMBuilder(LSTMDEPTH, PHRASEDIM, PATHLSTMDIM, model)
    cpathrevlstm = dy.LSTMBuilder(LSTMDEPTH, PHRASEDIM, PATHLSTMDIM, model)

    w_cp = model.add_parameters((PATHDIM, 2 * PATHLSTMDIM))
    b_cp = model.add_parameters((PATHDIM, 1))
###################################################
w_z = model.add_parameters((HIDDENDIM, ALL_FEATS_DIM))
b_z = model.add_parameters((HIDDENDIM, 1))
w_f = model.add_parameters((1, HIDDENDIM))
b_f = model.add_parameters((1, 1))
######################################################
if USE_CHV:
    ch_z = model.add_parameters((HIDDENCHDIM, ALL_FEATS_DIM))
    bch_z = model.add_parameters((HIDDENCHDIM, 1))
    ch_f = model.add_parameters((1, HIDDENCHDIM))
    bch_f = model.add_parameters((1, 1))


if USE_PTB_CONSTITS:
    w_c = model.add_parameters((2, LSTMDIM))
    b_c = model.add_parameters((2, 1))
    w_fb = model.add_parameters((LSTMDIM, 2 * LSTMDIM))
    b_fb = model.add_parameters((LSTMDIM, 1))
    DELTA = len(trainexamples) * 1.0 / len(ptbexamples)
    sys.stderr.write("weighing PTB down by %f\n" % DELTA)

#first hidden layer, given input layer
def get_base_embeddings(trainmode, unkdtokens, tg_start, sentence):
    #print(sentence)
    #print('this is unkd tokens    :', unkdtokens)
    #print('this is tg_start  :', tg_start)
    sentlen = len(unkdtokens)
    #embedding_x
    if trainmode:
        emb_x = [dy.noise(v_x[tok], 0.1) for tok in unkdtokens]
    else:
        emb_x = [v_x[tok] for tok in unkdtokens]
    pos_x = [p_x[pos] for pos in sentence.postags]
    dist_x = [dy.scalarInput(i - tg_start + 1) for i in xrange(sentlen)]
    #here you are giving as base input a matrix of vectors representing the sentence with embedding, part of speech,
    #and dist_x (don't know what this is, scalarInput) and telling it what position in the sentence each goes
    #this is the input layer
    baseinp_x = [(w_i * dy.concatenate([emb_x[j], pos_x[j], dist_x[j]]) + b_i) for j in xrange(sentlen)]

    if USE_WV:
        for j in xrange(sentlen):
            if unkdtokens[j] in wvs:
                nonupdatedwv = dy.nobackprop(e_x[unkdtokens[j]])
                baseinp_x[j] = baseinp_x[j] + w_e * nonupdatedwv + b_e

    embposdist_x = [dy.rectify(baseinp_x[j]) for j in xrange(sentlen)]
    #print(' this is embpostdist: ' , embposdist_x, '\n\n this is baseinp_x:  ', baseinp_x, '\n\n\n\n\n')
    if USE_DROPOUT:
        basefwdlstm.set_dropout(DROPOUT_RATE)
        baserevlstm.set_dropout(DROPOUT_RATE)
    bfinit = basefwdlstm.initial_state()
    basefwd = bfinit.transduce(embposdist_x)
    brinit = baserevlstm.initial_state()
    baserev = brinit.transduce(reversed(embposdist_x))
    #this is the first hidden layer, the green one with back and forth arrows
    basebi_x = [dy.rectify(w_bi * dy.concatenate([basefwd[eidx], baserev[sentlen - eidx - 1]]) +
                    b_bi) for eidx in xrange(sentlen)]

    if USE_DEPS:
        dhead_x = [embposdist_x[dephead] for dephead in sentence.depheads]
        dheadp_x = [pos_x[dephead] for dephead in sentence.depheads]
        drel_x = [dr_x[deprel] for deprel in sentence.deprels]
        baseinp_x = [dy.rectify(w_di * dy.concatenate([dhead_x[j], dheadp_x[j], drel_x[j], basebi_x[j]]) +
                        b_di) for j in xrange(sentlen)]
        basebi_x = baseinp_x

    return basebi_x
if USE_CHV:
    def get_base_character_embeddings(trainmode, unkdcharacters, tg_start):
        #print(sentence)
        #print('this is unkd tokens    :', unkdtokens)
        #print('this is tg_start  :', tg_start)
        sentlen = len(unkdcharacters)
        #embedding_x
        if trainmode:
            chemb_x = [dy.noise(ch_x[char], 0.1) for char in unkdcharacters]
        else:
            chemb_x = [ch_x[char] for char in unkdcharacters]
        dist_x = [dy.scalarInput(i - tg_start + 1) for i in xrange(sentlen)]
        #here you are giving as base input a matrix of vectors representing the sentence with embedding, part of speach,
        #and dist_x (don't know what this is, scalarInput) and telling it what position in the sentence each goes
        #this is the input layer
        basechinp_x = [(ch_i * dy.concatenate([chemb_x[j], dist_x[j]]) + bch_i) for j in xrange(sentlen)]

        if USE_CHV:
            for j in xrange(sentlen):
                if unkdcharacters[j] in chvs:
                    nonupdatedchv = dy.nobackprop(ch_x[unkdcharacters[j]])
                    basechinp_x[j] = basechinp_x[j] + ch_e * nonupdatedchv + bch_e

        chembdist_x = [dy.rectify(basechinp_x[j]) for j in xrange(sentlen)]
        #print(' this is embpostdist: ' , embposdist_x, '\n\n this is baseinp_x:  ', baseinp_x, '\n\n\n\n\n')
        if USE_DROPOUT:
            charfwdlstm.set_dropout(DROPOUT_RATE)
            charrevlstm.set_dropout(DROPOUT_RATE)
        bfinit = charfwdlstm.initial_state()
        basefwd = bfinit.transduce(chembdist_x)
        brinit = charrevlstm.initial_state()
        baserev = brinit.transduce(reversed(chembdist_x))
        #this is the first hidden layer, the green one with back and forth arrows
        basechbi_x = [dy.rectify(ch_bi * dy.concatenate([basefwd[eidx], baserev[sentlen - eidx - 1]]) +
                        bch_bi) for eidx in xrange(sentlen)]


        return basechbi_x


def get_target_frame_embeddings(embposdist_x, tfdict):
    tfkeys = sorted(tfdict)
    tg_start = tfkeys[0]
    sentlen = len(embposdist_x)

    # Adding target word feature
    lu, frame = tfdict[tg_start]
    tginit = tgtlstm.initial_state()
    target_x = tginit.transduce(embposdist_x[tg_start: tg_start + len(tfkeys) + 1])[-1]

    # Adding context features
    ctxt = range(tg_start - 1, tfkeys[-1] + 2)
    if ctxt[0] < 0: ctxt = ctxt[1:]
    if ctxt[-1] > sentlen: ctxt = ctxt[:-1]
    c_init = ctxtlstm.initial_state()
    ctxt_x = c_init.transduce(embposdist_x[ctxt[0]:ctxt[-1]])[-1]

    # Adding features specific to LU and frame
    lu_v = lu_x[lu.id]
    lp_v = lp_x[lu.posid]

    if USE_HIER and frame.id in frmrelmap:
        #print('hi, frameid  :', frame.id)
        frame_v = dy.esum([frm_x[frame.id]] + [frm_x[par] for par in frmrelmap[frame.id]])
    else:
        frame_v = frm_x[frame.id]
    tfemb = dy.concatenate([lu_v, lp_v, frame_v, target_x, ctxt_x])
    return tfemb, frame
if USE_CHV:
    def get_target_frame_character_embeddings(embposdist_x, tfdict):
        tfkeys = sorted(tfdict)
        tg_start = tfkeys[0]
        sentlen = len(embposdist_x)

        # Adding target word feature
        lu, frame = tfdict[tg_start]
        tginit = chtgtlstm.initial_state()
        target_x = tginit.transduce(embposdist_x[tg_start: tg_start + len(tfkeys) + 1])[-1]

        # Adding context features
        ctxt = range(tg_start - 1, tfkeys[-1] + 2)
        if ctxt[0] < 0: ctxt = ctxt[1:]
        if ctxt[-1] > sentlen: ctxt = ctxt[:-1]
        c_init = chctxtlstm.initial_state()
        ctxt_x = c_init.transduce(embposdist_x[ctxt[0]:ctxt[-1]])[-1]

        # Adding features specific to LU and frame
        lu_v = lu_x[lu.id]
        lp_v = lp_x[lu.posid]

        if USE_HIER and frame.id in frmrelmap:
            #print('hi, frameid  :', frame.id)
            frame_v = dy.esum([frm_x[frame.id]] + [frm_x[par] for par in frmrelmap[frame.id]])
        else:
            frame_v = frm_x[frame.id]
        tfemb = dy.concatenate([lu_v, lp_v, frame_v, target_x, ctxt_x])

        return tfemb, frame

def get_span_embeddings(embpos_x):
    sentlen = len(embpos_x)
    fws = [[None for _ in xrange(sentlen)] for _ in xrange(sentlen)]
    bws = [[None for _ in xrange(sentlen)] for _ in xrange(sentlen)]

    if USE_DROPOUT:
        builders[0].set_dropout(DROPOUT_RATE)
        builders[1].set_dropout(DROPOUT_RATE)

    for i in xrange(sentlen):
        fw_init = builders[0].initial_state()
        tmpfws = fw_init.transduce(embpos_x[i:])
        if len(tmpfws) != sentlen - i:
            raise Exception("incorrect number of forwards", len(tmpfws), i, sentlen)

        spanend = sentlen
        if USE_SPAN_CLIP: spanend = min(sentlen, i + ALLOWED_SPANLEN + 1)
        for j in xrange(i, spanend):
            # for j in xrange(i, sentlen):
            fws[i][j] = tmpfws[j - i]

        bw_init = builders[1].initial_state()
        tmpbws = bw_init.transduce(reversed(embpos_x[:i + 1]))
        if len(tmpbws) != i + 1:
            raise Exception("incorrect number of backwards", i, len(tmpbws))
        spansize = i + 1
        if USE_SPAN_CLIP and spansize - 1 > ALLOWED_SPANLEN:
            spansize = ALLOWED_SPANLEN + 1
        for k in xrange(spansize):
            bws[i - k][i] = tmpbws[k]

    return fws, bws

def get_deppath_embeddings(sentence, embpos_x):
    spaths = {}
    for spath in set(sentence.shortest_paths.values()):
        shp = [embpos_x[node] for node in spath]
        if USE_DROPOUT:
            pathfwdlstm.set_dropout(DROPOUT_RATE)
            pathrevlstm.set_dropout(DROPOUT_RATE)
        pfinit = pathfwdlstm.initial_state()
        pathfwd = pfinit.transduce(shp)
        prinit = pathrevlstm.initial_state()
        pathrev = prinit.transduce(reversed(shp))

        pathlstm = dy.rectify(w_p * dy.concatenate([pathfwd[-1], pathrev[-1]]) + b_p)

        spaths[spath] = pathlstm
    return spaths


def get_cpath_embeddings(sentence):
    phrpaths = {}
    for phrpath in set(sentence.cpaths.values()):
        shp = [ct_x[node] for node in phrpath]
        if USE_DROPOUT:
            cpathfwdlstm.set_dropout(DROPOUT_RATE)
            cpathrevlstm.set_dropout(DROPOUT_RATE)
        cpfinit = cpathfwdlstm.initial_state()
        cpathfwd = cpfinit.transduce(shp)
        cprinit = cpathrevlstm.initial_state()
        cpathrev = cprinit.transduce(reversed(shp))

        cpathlstm = dy.rectify(w_cp * dy.concatenate([cpathfwd[-1], cpathrev[-1]]) + b_cp)

        phrpaths[phrpath] = cpathlstm
    return phrpaths


def get_factor_expressions(fws, bws, tfemb, tfdict, valid_fes, sentence, spaths_x=None, cpaths_x=None):
    factexprs = {}
    sentlen = len(fws)

    sortedtfd = sorted(tfdict.keys())
    targetspan = (sortedtfd[0], sortedtfd[-1])

    for j in xrange(sentlen):
        istart = 0
        if USE_SPAN_CLIP and j > ALLOWED_SPANLEN: istart = max(0, j - ALLOWED_SPANLEN)
        for i in xrange(istart, j + 1):

            spanlen = dy.scalarInput(j - i + 1)
            logspanlen = dy.scalarInput(math.log(j - i + 1))
            spanwidth = sp_x[SpanWidth.howlongisspan(i, j)]
            spanpos = ap_x[ArgPosition.whereisarg((i, j), targetspan)]

            fbemb_ij = dy.concatenate([fws[i][j], bws[i][j], tfemb, spanlen, logspanlen, spanwidth, spanpos])


            if USE_DEPS:
                outs = oh_s[OutHeads.getnumouts(i, j, sentence.outheads)]
                shp = spaths_x[sentence.shortest_paths[(i, j, targetspan[0])]]
                fbemb_ij = dy.concatenate([fbemb_ij_basic, outs, shp])
            elif USE_CONSTITS:
                isconstit = dy.scalarInput((i, j) in sentence.constitspans)
                lca = ct_x[sentence.lca[(i, j)][1]]
                phrp = cpaths_x[sentence.cpaths[(i, j, targetspan[0])]]
                fbemb_ij = dy.concatenate([fbemb_ij_basic, isconstit, lca, phrp])

            for y in valid_fes:
                fctr = Factor(i, j, y)
                if USE_HIER and y in feparents:
                    fefixed = dy.esum([fe_x[y]] + [fe_x[par] for par in feparents[y]])
                else:
                    fefixed = fe_x[y]
                fbemb_ijy = dy.concatenate([fefixed, fbemb_ij])
                factexprs[fctr] = w_f * dy.rectify(w_z * fbemb_ijy + b_z) + b_f
    return factexprs


def hamming_cost(factor, goldfactors):
    if factor in goldfactors:
        return dy.scalarInput(0)
    return dy.scalarInput(1)


def recall_oriented_cost(factor, goldfactors):
    alpha = RECALL_ORIENTED_COST
    beta = 1

    if factor in goldfactors:
        return dy.scalarInput(0)
    i = factor.begin
    j = factor.end
    alphabetacost = 0
    if factor.label != NOTANFEID:
        alphabetacost += beta
    # find number of good gold factors it kicks out
    for gf in goldfactors:
        if i <= gf.begin <= j and gf.label != NOTANFEID:
            alphabetacost += alpha

    return dy.scalarInput(alphabetacost)


def cost(factor, goldfactors):
    if options.cost == "hamming":
        return hamming_cost(factor, goldfactors)
    elif options.cost == "recall":
        return recall_oriented_cost(factor, goldfactors)
    else:
        raise Exception("undefined cost type", options.cost)


def get_logloss_partition(factorexprs, valid_fes, sentlen):
    logalpha = [None for _ in xrange(sentlen)]
    # ssum = lossformula(sentlen, len(valid_fes))
    for j in xrange(sentlen):
        # full length spans
        spanscores = []
        if not USE_SPAN_CLIP or j <= ALLOWED_SPANLEN:
            spanscores = [factorexprs[Factor(0, j, y)] for y in valid_fes]
        # recursive case
        istart = 0
        if USE_SPAN_CLIP and j > ALLOWED_SPANLEN: istart = max(0, j - ALLOWED_SPANLEN - 1)
        for i in xrange(istart, j):
            facscores = [logalpha[i] + factorexprs[Factor(i + 1, j, y)] for y in valid_fes]
            spanscores.extend(facscores)

        if not USE_SPAN_CLIP and len(spanscores) != len(valid_fes) * (j + 1):
            raise Exception("counting errors")
        logalpha[j] = dy.logsumexp(spanscores)

    return logalpha[sentlen - 1]


def get_softmax_margin_partition(factorexprs, goldfactors, valid_fes, sentlen):
    logalpha = [None for _ in xrange(sentlen)]
    for j in xrange(sentlen):
        # full length spans
        spanscores = []
        if not USE_SPAN_CLIP or j <= ALLOWED_SPANLEN:
            spanscores = [factorexprs[Factor(0, j, y)]
                          + cost(Factor(0, j, y), goldfactors) for y in valid_fes]

        # recursive case
        istart = 0
        if USE_SPAN_CLIP and j > ALLOWED_SPANLEN: istart = max(0, j - ALLOWED_SPANLEN - 1)
        for i in xrange(istart, j):
            facscores = [logalpha[i]
                         + factorexprs[Factor(i + 1, j, y)]
                         + cost(Factor(i + 1, j, y), goldfactors) for y in valid_fes]
            spanscores.extend(facscores)

        if not USE_SPAN_CLIP and len(spanscores) != len(valid_fes) * (j + 1):
            raise Exception("counting errors")
        logalpha[j] = dy.logsumexp(spanscores)

    return logalpha[sentlen - 1]


def get_hinge_partition(factorexprs, goldfacs, valid_fes, sentlen):
    alpha = [None for _ in xrange(sentlen)]
    backpointers = [None for _ in xrange(sentlen)]

    for j in xrange(sentlen):
        # full length spans
        bestscore = float("-inf")
        if not USE_SPAN_CLIP or j <= ALLOWED_SPANLEN:
            for y in valid_fes:
                factor = Factor(0, j, y)
                facscore = factorexprs[factor] + cost(factor, goldfacs)
                if facscore.scalar_value() > bestscore:
                    bestscore = facscore.scalar_value()
                    alpha[j] = facscore
                    backpointers[j] = (0, y)

        # recursive case
        istart = 0
        if USE_SPAN_CLIP and j > ALLOWED_SPANLEN: istart = max(0, j - ALLOWED_SPANLEN - 1)
        for i in xrange(istart, j):
            for y in valid_fes:
                factor = Factor(i + 1, j, y)
                facscore = alpha[i] + factorexprs[factor] + cost(factor, goldfacs)
                if facscore.scalar_value() > bestscore:
                    bestscore = facscore.scalar_value()
                    alpha[j] = facscore
                    backpointers[j] = (i + 1, y)

    predfactors = []
    j = sentlen - 1
    i = backpointers[j][0]
    while i >= 0:
        fe = backpointers[j][1]
        predfactors.append(Factor(i, j, fe))
        if i == 0:
            break
        j = i - 1
        i = backpointers[j][0]
    return alpha[sentlen - 1], predfactors


def get_hinge_loss(factorexprs, gold_fes, valid_fes, sentlen):
    goldfactors = [Factor(span[0], span[1], feid) for feid in gold_fes for span in gold_fes[feid]]
    numeratorexprs = [factorexprs[gf] for gf in goldfactors]
    numerator = dy.esum(numeratorexprs)

    denominator, predfactors = get_hinge_partition(factorexprs, goldfactors, valid_fes, sentlen)

    if set(predfactors) == set(goldfactors):
        return None

    hingeloss = denominator - numerator
    if denominator.scalar_value() < numerator.scalar_value():
        raise Exception("ERROR: predicted cost less than gold!",
                        denominator.scalar_value(),
                        numerator.scalar_value(),
                        hingeloss.scalar_value())
    return hingeloss


def get_constit_loss(fws, bws, goldspans):
    if not USE_PTB_CONSTITS:
        raise Exception("should not be using the constit loss now!", USE_PTB_CONSTITS)

    if len(goldspans) == 0:
        return None, 0

    losses = []
    sentlen = len(fws)

    for j in xrange(sentlen):
        istart = 0
        if USE_SPAN_CLIP and j > ALLOWED_SPANLEN: istart = max(0, j - ALLOWED_SPANLEN)
        for i in xrange(istart, j + 1):
            constit_ij = w_c * dy.rectify(w_fb * dy.concatenate([fws[i][j], bws[i][j]]) + b_fb) + b_c
            logloss = log_softmax(constit_ij)

            isconstit = int((i, j) in goldspans)
            losses.append(pick(logloss, isconstit))

    ptbconstitloss = dy.scalarInput(DELTA) * -dy.esum(losses)
    numspanstagged = len(losses)
    return ptbconstitloss, numspanstagged


def get_loss(factorexprs, gold_fes, valid_fes, sentlen):
    # if options.loss == "hinge":
    #     return get_hinge_loss(factorexprs, gold_fes, valid_fes, sentlen)

    goldfactors = [Factor(span[0], span[1], feid) for feid in gold_fes for span in gold_fes[feid]]
    #print([key.to_str(FEDICT) for key in factorexprs.keys()])
    #print([gf.to_str(FEDICT) for gf in goldfactors])
    #for gf in goldfactors:
        #print(gf.to_str(FEDICT))
        #print('factorexprs: ', factorexprs[gf])
    numeratorexprs = [factorexprs[gf] for gf in goldfactors]
    numerator = dy.esum(numeratorexprs)

    # if options.loss == "log":
    #     partition = get_logloss_partition(factorexprs, valid_fes, sentlen)
    if options.loss == "softmaxm":
        partition = get_softmax_margin_partition(factorexprs, goldfactors, valid_fes, sentlen)
    else:
        raise Exception("undefined loss function", options.loss)

    lossexp = partition - numerator
    if partition.scalar_value() < numerator.scalar_value():
        sys.stderr.write("WARNING: partition ~~ numerator! possibly overfitting difference = %f\n"
                         % lossexp.scalar_value())
        return None

    if lossexp.scalar_value() < 0.0:
        sys.stderr.write(str(gold_fes) + "\ngolds\n")
        gsum = 0
        for fac in goldfactors:
            gsum += factorexprs[fac].scalar_value()
            sys.stderr.write(fac.to_str(FEDICT) + " " + str(factorexprs[fac].scalar_value()) + "\n")
        sys.stderr.write("my calculation = " + str(gsum) + " vs " + str(numerator.scalar_value()) + "\n")
        for j in xrange(sentlen):
            sys.stderr.write(":" + str(j) + "\t")
            if not USE_SPAN_CLIP or j <= ALLOWED_SPANLEN:
                sys.stderr.write("0 ")
            istart = 0
            if USE_SPAN_CLIP and j > ALLOWED_SPANLEN: istart = max(0, j - ALLOWED_SPANLEN - 1)
            for i in xrange(istart, j):
                sys.stderr.write(str(i + 1) + " ")
            sys.stderr.write("\n")
        raise Exception("negative probability! probably overcounting spans?",
                        numerator.scalar_value(),
                        partition.scalar_value(),
                        lossexp.scalar_value())
    return lossexp


def decode(factexprscalars, sentlen, valid_fes):
    alpha = [None for _ in xrange(sentlen)]
    backpointers = [None for _ in xrange(sentlen)]
    if USE_DROPOUT:
        raise Exception("incorrect usage of dropout, turn off!\n")

    for j in xrange(sentlen):
        if USE_SPAN_CLIP and j > ALLOWED_SPANLEN: continue
        bestscore = float("-inf")
        bestlabel = None
        for y in valid_fes:
            fac = Factor(0, j, y)
            facscore = math.exp(factexprscalars[fac])
            if facscore > bestscore:
                bestscore = facscore
                bestlabel = y
        alpha[j] = bestscore
        backpointers[j] = (0, bestlabel)

    for j in xrange(sentlen):
        bestscore = float("-inf")
        bestbeg = bestlabel = None
        if alpha[j] is not None:
            bestscore = alpha[j]
            bestbeg, bestlabel = backpointers[j]

        istart = 0
        if USE_SPAN_CLIP and j > ALLOWED_SPANLEN: istart = max(0, j - ALLOWED_SPANLEN - 1)
        for i in xrange(istart, j):
            for y in valid_fes:
                fac = Factor(i + 1, j, y)
                facscore = math.exp(factexprscalars[fac])
                if facscore * alpha[i] > bestscore:
                    bestscore = facscore * alpha[i]
                    bestlabel = y
                    bestbeg = i + 1
        alpha[j] = bestscore
        backpointers[j] = (bestbeg, bestlabel)

    j = sentlen - 1
    i = backpointers[j][0]
    argmax = {}
    while i >= 0:
        fe = backpointers[j][1]
        if fe in argmax:
            argmax[fe].append((i, j))
        else:
            argmax[fe] = [(i, j)]
        if i == 0:
            break
        j = i - 1
        i = backpointers[j][0]

    # merging neighboring spans in prediction (to combat spurious ambiguity)
    mergedargmax = {}
    for fe in argmax:
        mergedargmax[fe] = []
        if fe == NOTANFEID:
            mergedargmax[fe].extend(argmax[fe])
            continue

        argmax[fe].sort()
        mergedspans = [argmax[fe][0]]
        for span in argmax[fe][1:]:
            prevsp = mergedspans[-1]
            if span[0] == prevsp[1] + 1:
                prevsp = mergedspans.pop()
                mergedspans.append((prevsp[0], span[1]))
            else:
                mergedspans.append(span)
        mergedargmax[fe] = mergedspans
    return mergedargmax


def identify_fes(unkdtoks, sentence, tfdict, unkdchars=None, goldfes=None, testidx=None):
    dy.renew_cg()
    trainmode = (goldfes is not None)

    global USE_DROPOUT
    USE_DROPOUT = trainmode

    sentlen = len(unkdtoks)
    tfkeys = sorted(tfdict)
    tg_start = tfkeys[0]

    embpos_x = get_base_embeddings(trainmode, unkdtoks, tg_start, sentence)

    if unkdchars:
        embchars_x = get_base_character_embeddings(trainmode, unkdchars, tg_start)
        tfemb, frame = get_target_frame_character_embeddings(embchars_x, tfdict)
    else:
        tfemb, frame = get_target_frame_embeddings(embpos_x, tfdict)
    fws, bws = get_span_embeddings(embpos_x)
    valid_fes = frmfemap[frame.id] + [NOTANFEID]
    if USE_DEPS:
        spaths_x = get_deppath_embeddings(sentence, embpos_x)
        factor_exprs = get_factor_expressions(fws, bws, tfemb, tfdict, valid_fes, sentence, spaths_x=spaths_x)
    elif USE_CONSTITS:
        cpaths_x = get_cpath_embeddings(sentence)
        factor_exprs = get_factor_expressions(fws, bws, tfemb, tfdict, valid_fes, sentence, cpaths_x=cpaths_x)
    else:
        factor_exprs = get_factor_expressions(fws, bws, tfemb, tfdict, valid_fes, sentence)

    if trainmode:
        segrnnloss = get_loss(factor_exprs, goldfes, valid_fes, sentlen)
        if USE_PTB_CONSTITS:
            goldspans = []
            for feid in goldfes:
                if feid == NOTANFEID: continue
                goldspans.extend(goldfes[feid])

            constitloss, numspans = get_constit_loss(cfws, bws, goldspans)
            if segrnnloss is not None and constitloss is not None:
                # segrnnloss of 1 segmentation vs all, globally normalized
                return segrnnloss + constitloss, 1 + numspans
            elif segrnnloss is None:
                return constitloss, numspans
        return segrnnloss, 1  # segrnnloss of 1 segmentation vs all, globally normalized
    else:
        if SAVE_FOR_ENSEMBLE:
            outensapf = open(out_ens_file, "a")
            for fact in factor_exprs:
                outensapf.write(
                    str(testidx) + "\t"
                    + fact.to_str(FEDICT) + "\t"
                    + str((factor_exprs[fact]).scalar_value())
                    + "\n")
            outensapf.close()
        facexprscalars = {fact: factor_exprs[fact].scalar_value() for fact in factor_exprs}
        argmax = decode(facexprscalars, sentlen, valid_fes)
        return argmax


def identify_spans(unkdtoks, sentence, goldspans):
    renew_cg()

    embpos_x = get_base_embeddings(True, unkdtoks, 0, sentence)
    fws, bws = get_span_embeddings(embpos_x)

    return get_constit_loss(fws, bws, goldspans)


def print_as_conll(golds, pred_targmaps):
    with codecs.open(out_conll_file, "w", "utf-8") as f:
        for gold, pred in zip(golds, pred_targmaps):
            result = gold.get_str(predictedfes=pred)
            f.write(result + "\n")
        f.close()


def print_eval_result(examples, expredictions, logger):
    evalstarttime = time.time()
    corp_up, corp_ur, corp_uf, \
    corp_lp, corp_lr, corp_lf, \
    corp_wp, corp_wr, corp_wf, \
    corp_ures, corp_labldres, corp_tokres = evaluate_corpus_argid(
        examples, expredictions, corefrmfemap, NOTANFEID, logger)

    sys.stderr.write("\n[test] wpr = %.5f (%.1f/%.1f) wre = %.5f (%.1f/%.1f)\n"
                     "[test] upr = %.5f (%.1f/%.1f) ure = %.5f (%.1f/%.1f)\n"
                     "[test] lpr = %.5f (%.1f/%.1f) lre = %.5f (%.1f/%.1f)\n"
                     "[test] wf1 = %.5f uf1 = %.5f lf1 = %.5f [took %.3fs]\n"
                     % (corp_wp, corp_tokres[0], corp_tokres[1] + corp_tokres[0],
                        corp_wr, corp_tokres[0], corp_tokres[-1] + corp_tokres[0],
                        corp_up, corp_ures[0], corp_ures[1] + corp_ures[0],
                        corp_ur, corp_ures[0], corp_ures[-1] + corp_ures[0],
                        corp_lp, corp_labldres[0], corp_labldres[1] + corp_labldres[0],
                        corp_lr, corp_labldres[0], corp_labldres[-1] + corp_labldres[0],
                        corp_wf, corp_uf, corp_lf,
                        time.time() - evalstarttime))


logger = open("{}/argid-prediction-analysis.log".format(model_dir), "w")

if options.mode in ["test", "refresh", "predict"]:
    sys.stderr.write("Reloading model from {} ...\n".format(model_file_name))
    model.populate(model_file_name)

best_dev_f1 = 0.0
if options.mode in ["refresh"]:
    with open(os.path.join(model_dir, "best-dev-f1.txt"), "r") as fin:
        for line in fin:
            best_dev_f1 = float(line.strip())
    fin.close()
    sys.stderr.write("Best dev F1 so far = %.4f\n" % best_dev_f1)

if options.mode in ["train", "refresh"]:
    loss = 0.0
    last_updated_epoch = 0

    if USE_PTB_CONSTITS:
        trainexamples = trainexamples + ptbexamples

    starttime = time.time()
    for epoch in xrange(NUMEPOCHS):
        random.shuffle(trainexamples)

        for idx, trex in enumerate(trainexamples, 1):
            if (idx - 1) % LOSS_EVAL_EPOCH == 0 and idx > 1:
                adam.status()
                sys.stderr.write("epoch=%d.%d loss=%.4f [took %.3fs]\n" % (
                    epoch, idx-1, (loss/idx), time.time() - starttime))
                starttime = time.time()

            unkedtoks = []
            unk_replace_tokens(trex.tokens, unkedtoks, VOCDICT, UNK_PROB, UNKTOKEN)
            if USE_CHV:
                unkedchars = []
                unk_replace_characters(trex.chars, unkedchars, CHARDICT, UNK_PROB, UNKCHAR, CHARPAD, CHARSPACE)


            if USE_PTB_CONSTITS and type(trex) == Sentence:  # a PTB example
                trexloss, taggedinex = identify_spans(unkedtoks,
                                                      trex,
                                                      trex.constitspans.keys())
            else:  # an FN example
                if USE_CHV:
                    trexloss, taggedinex = identify_fes(unkedtoks,
                                                        trex.sentence,
                                                        trex.targetframedict,
                                                        unkdchars=unkedchars,
                                                        goldfes=trex.invertedfes)
                else:
                    trexloss, taggedinex = identify_fes(unkedtoks,
                                                        trex.sentence,
                                                        trex.targetframedict,
                                                        goldfes=trex.invertedfes)
            # tagged += taggedinex

            if trexloss is not None:
                loss += trexloss.scalar_value()
                trexloss.backward()
                adam.update()

            if (idx - 1) % DEV_EVAL_EPOCHS == 0 and idx > 1:
                devstarttime = time.time()
                ures = labldres = tokenwise = [0.0, 0.0, 0.0]
                predictions = []

                for devex in devexamples:
                    if USE_CHV:
                        dargmax = identify_fes(devex.tokens,
                                               devex.sentence,
                                               devex.targetframedict,
                                               unkdchars=devex.chars)
                    else:
                        dargmax = identify_fes(devex.tokens,
                                               devex.sentence,
                                               devex.targetframedict)
                    if devex.frame.id in corefrmfemap:
                        corefes = corefrmfemap[devex.frame.id]
                    else:
                        corefes = {}
                    u, l, t = evaluate_example_argid(devex.invertedfes, dargmax, corefes, len(devex.tokens), NOTANFEID)
                    ures = np.add(ures, u)
                    labldres = np.add(labldres, l)
                    tokenwise = np.add(tokenwise, t)

                    predictions.append(dargmax)

                up, ur, uf = calc_f(ures)
                lp, lr, lf = calc_f(labldres)
                wp, wr, wf = calc_f(tokenwise)
                sys.stderr.write("[dev epoch=%d.%d] wprec = %.5f wrec = %.5f wf1 = %.5f\n"
                                 "[dev epoch=%d.%d] uprec = %.5f urec = %.5f uf1 = %.5f\n"
                                 "[dev epoch=%d.%d] lprec = %.5f lrec = %.5f lf1 = %.5f"
                                 % (epoch, idx, wp, wr, wf, epoch, idx, up, ur, uf, epoch, idx, lp, lr, lf))

                if lf > best_dev_f1:
                    best_dev_f1 = lf
                    with open(os.path.join(model_dir, "best-dev-f1.txt"), "w") as fout:
                        fout.write("{}\n".format(best_dev_f1))
                        fout.close()

                    print_as_conll(devexamples, predictions)
                    sys.stderr.write(" -- saving to {}".format(model_file_name))
                    model.save(model_file_name)
                    last_updated_epoch = epoch
                sys.stderr.write(" [took %.3fs]\n" % (time.time() - devstarttime))
                starttime = time.time()
        if epoch - last_updated_epoch > PATIENCE:
            sys.stderr.write("Ran out of patience, ending training.\n")
            break
        loss = 0.0

elif options.mode == "ensemble":
    exfs = {x: {} for x in xrange(len(devexamples))}
    USE_DROPOUT = False

    sys.stderr.write("reading ensemble factors...")
    enf = open(in_ens_file, "rb")
    for l in enf:
        fields = l.split("\t")
        fac = Factor(int(fields[1]), int(fields[2]), FEDICT.getid(fields[3]))
        exfs[int(fields[0])][fac] = float(fields[4])
    enf.close()

    sys.stderr.write("done!\n")
    teststarttime = time.time()
    sys.stderr.write("testing " + str(len(devexamples)) + " examples ...\n")

    testpredictions = []
    for tidx, testex in enumerate(devexamples, 1):
        if tidx % 100 == 0:
            sys.stderr.write(str(tidx) + "...")
        valid_fes_for_frame = frmfemap[testex.frame.id] + [NOTANFEID]
        testargmax = decode(exfs[tidx - 1], len(testex.tokens), valid_fes_for_frame)
        testpredictions.append(testargmax)

    sys.stderr.write(" [took %.3fs]\n" % (time.time() - teststarttime))
    sys.stderr.write("printing output conll to " + out_conll_file + " ... ")
    print_as_conll(devexamples, testpredictions)
    sys.stderr.write("done!\n")
    print_eval_result(devexamples, testpredictions, logger)
    sys.stderr.write("printing frame-elements to " + fe_file + " ...\n")
    convert_conll_to_frame_elements(out_conll_file, fe_file)
    sys.stderr.write("done!\n")

elif options.mode == "test":
    if SAVE_FOR_ENSEMBLE:
        outensf = open(out_ens_file, "w")
        outensf.close()

    sys.stderr.write("testing " + str(len(devexamples)) + " examples ...\n")
    teststarttime = time.time()

    testpredictions = []
    for tidx, testex in enumerate(devexamples, 1):
        if tidx % 100 == 0:
            sys.stderr.write(str(tidx) + "...")
        if USE_CHV:
            testargmax = identify_fes(testex.tokens,
                                      testex.sentence,
                                      testex.targetframedict,
                                      testidx=tidx - 1, unkdchars=testex.chars)
        else:
            testargmax = identify_fes(testex.tokens,
                                      testex.sentence,
                                      testex.targetframedict,
                                      testidx=tidx - 1)
        testpredictions.append(testargmax)

    sys.stderr.write(" [took %.3fs]\n" % (time.time() - teststarttime))
    sys.stderr.write("printing output conll to " + out_conll_file + " ... ")
    print_as_conll(devexamples, testpredictions)
    sys.stderr.write("done!\n")
    print_eval_result(devexamples, testpredictions, logger)
    sys.stderr.write("printing frame-elements to " + fe_file + " ...\n")
    convert_conll_to_frame_elements(out_conll_file, fe_file)
    sys.stderr.write("done!\n")
    logger.close()

elif options.mode == "predict":
    predictions = []
    for instance in instances:
        if USE_CHV:
            prediction = identify_fes(instance.tokens,
                                      instance.sentence,
                                      instance.targetframedict,
                                      unkdchars=instance.chars)
        else:
            prediction = identify_fes(instance.tokens,
                                      instance.sentence,
                                      instance.targetframedict)
        predictions.append(prediction)
    sys.stderr.write("Printing output in CoNLL format to {}\n".format(out_conll_file))
    print_as_conll(instances, predictions)
    sys.stderr.write("Done!\n")