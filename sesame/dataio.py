# -*- coding: utf-8 -*-
import codecs
import os
import sys
import tarfile
import xml.etree.ElementTree as et
import numpy as np

from nltk.corpus import BracketParseCorpusReader
from sklearn.decomposition import PCA

from conll09 import *
from globalconfig import *
from sentence import *

def analyze_constits_fes(examples):
    matchspan = 0.0
    notmatch = 0.0
    matchph = {}
    for ex in examples:
        for fe in ex.invertedfes:
            if fe == FEDICT.getid(EMPTY_LABEL):
                continue
            for span in ex.invertedfes[fe]:
                if span in ex.sentence.constitspans:
                    matchspan += 1
                    phrases = ex.sentence.constitspans[span]
                    for phrase in phrases:
                        if phrase not in matchph:
                            matchph[phrase] = 0
                        matchph[phrase] += 1
                else:
                    notmatch += 1
    tot = matchspan + notmatch
    sys.stderr.write("matches = %d %.2f%%\n"
                     "non-matches = %d %.2f%%\n"
                     "total = %d\n"
                     % (matchspan, matchspan*100/tot, notmatch, notmatch*100/tot, tot))
    sys.stderr.write("phrases which are constits = %d\n" %(len(matchph)))

def read_conll(conll_file, syn_type=None):
    sys.stderr.write("\nReading {} ...\n".format(conll_file))

    read_depsyn = read_constits = False
    if syn_type == "dep":
        read_depsyn = True
    elif syn_type == "constit":
        read_constits = True
        cparses = read_brackets(CONSTIT_MAP[conll_file])


    examples = []
    elements = []
    missingargs = 0.0
    totalexamples = 0.0

    next_ex = 0
    with codecs.open(conll_file, "r", "utf-8") as cf:
        snum = -1
        for l in cf:
            l = l.strip()
            if l == "":
                if elements[0].sent_num != snum:
                    sentence = Sentence(syn_type, elements=elements)
                    if read_constits:
                        sentence.get_all_parts_of_ctree(cparses[next], CLABELDICT, True)
                    next_ex += 1
                    snum = elements[0].sent_num
                e = CoNLL09Example(sentence, elements)
                examples.append(e)
                if read_depsyn:
                    sentence.get_all_paths_to(sorted(e.targetframedict.keys())[0])
                elif read_constits:
                    sentence.get_cpath_to_target(sorted(e.targetframedict.keys())[0])

                if e.numargs == 0:
                    missingargs += 1

                totalexamples += 1

                elements = []
                continue
            else:
                #print(l)
                elements.append(CoNLL09Element(l, read_depsyn))
        cf.close()
    sys.stderr.write("# examples in %s : %d in %d sents\n" %(conll_file, len(examples), next_ex))
    sys.stderr.write("# examples with missing arguments : %d\n" %missingargs)
    if read_constits:
        analyze_constits_fes(examples)
    return examples, missingargs, totalexamples

def create_target_lu_map():
    multiplicity = 0
    repeated = 0
    total = 0

    target_lu_map = {}
    lu_names = set([])
    for frame_file in os.listdir(FRAME_DIR):
        with open(FRAME_DIR+'/'+frame_file, 'r') as f:
            for line in f:
                line = line.split()
                if line[0] == 'LUs':
                    for lu_name in line[1:]:
                        lu_names.add(lu_name)
                        target_name = lu_name.split('.')[0]
                        LUDICT.addstr(target_name)
                        if target_name not in target_lu_map:
                            target_lu_map[target_name] = set([])
                        else:
                            repeated += 1
                        target_lu_map[target_name].add(lu_name)
                        if len(target_lu_map[target_name]) > multiplicity:
                            multiplicity = len(target_lu_map[target_name])
                        total += 1

    sys.stderr.write("# unique targets = {}\n".format(len(target_lu_map)))
    sys.stderr.write("# total targets = {}\n".format(total))
    sys.stderr.write("# targets with multiple LUs = {}\n".format(repeated))
    sys.stderr.write("# max LUs per target = {}\n\n".format(multiplicity))
    return target_lu_map, lu_names


def read_fes_lus(frame_file):
    #opens a frame file to parse xml
    f = open(frame_file, "r")
    frcount = 0
    #for each "frame" in the frame file, get the frame name and put it in frame dict
    lus = []
    fes = []
    corefes = []
    for line in f:
        line_items = line.strip().split()
        if line_items[0] == 'Frame':
            framename = line_items[1]
            frid = FRAMEDICT.addstr(framename)
            frcount += 1
        #now you parse through the frame elements and get their names and append to FEDICT
        elif line_items[0] == 'FEs':
            for fe in line_items[1:]:
                fename = fe
                feid = FEDICT.addstr(fename)
                fes.append(feid)
                corefes.append(feid)
        elif line_items[0] == 'LUs':
            #now for each LU associated with the frame, add it to the dict, split the . though and add LU to LUDICT and LUPOS to LUPOSDICT
            for lu in line_items[1:]:
                lu_fields = lu.split(".")
                luid = LUDICT.addstr(lu_fields[0])
                LUPOSDICT.addstr(lu_fields[1])
                lus.append(luid)
    f.close()
    #fr is a number like 1047, fes is a list of numbers, corefes is a list of numbers, lus is a list of numbers
    return frid, fes, corefes, lus


def read_frame_maps():
    sys.stderr.write("\nReading the frame-element - frame map from {} ...\n".format(FRAME_DIR))

    frmfemap = {}
    corefrmfemap = {}
    lufrmmap = {}
    maxfesforframe = 0
    longestframe = None

    for f in os.listdir(FRAME_DIR):
        framef = os.path.join(FRAME_DIR, f)
        if framef.endswith("xsl"):
            continue
        frm, fes, corefes, lus = read_fes_lus(framef)
        frmfemap[frm] = fes
        corefrmfemap[frm] = corefes
        if len(frmfemap[frm]) > maxfesforframe:
            maxfesforframe = len(frmfemap[frm])
            longestframe = frm
        for l in lus:
            if l not in lufrmmap:
                lufrmmap[l] = []
            lufrmmap[l].append(frm)

    sys.stderr.write("Max! {} frame-elements for frame: {}\n\n".format(maxfesforframe, FRAMEDICT.getstr(longestframe)))
    return frmfemap, corefrmfemap, lufrmmap


def read_related_lus():
    sys.stderr.write("\nReading the frame-LU map from " + FRAME_DIR + " ...\n")

    lu_to_frame_dict = {}
    tot_frames = 0.
    max_frames = 0
    tot_lus = 0.
    longestlu = None

    frame_to_lu_dict = {}
    max_lus = 0
    longestfrm = None

    for f in os.listdir(FRAME_DIR):
        framef = os.path.join(FRAME_DIR, f)
        if framef.endswith("xsl"):
            continue
        tot_frames += 1
        frm, fes, corefes, lus = read_fes_lus(framef)
        #print("here is what you need to make:   ", frm, fes, corefes, lus)
        for l in lus:
            tot_lus += 1
            if l not in lu_to_frame_dict:
                lu_to_frame_dict[l] = set([])
            lu_to_frame_dict[l].add(frm)
            if len(lu_to_frame_dict[l]) > max_frames:
                max_frames = len(lu_to_frame_dict[l])
                longestlu = l


            if frm not in frame_to_lu_dict:
                frame_to_lu_dict[frm] = set([])
            frame_to_lu_dict[frm].add(l)
            if len(frame_to_lu_dict[frm]) > max_lus:
                max_lus = len(frame_to_lu_dict[frm])
                longestfrm = frm

    related_lus = {}
    for l in lu_to_frame_dict:
        for frm in lu_to_frame_dict[l]:
            if frm in frame_to_lu_dict:
                if l not in related_lus:
                    related_lus[l] = set([])
                related_lus[l].update(frame_to_lu_dict[frm])

    tot_frames_per_lu = sum([len(lu_to_frame_dict[l]) for l in lu_to_frame_dict])
    avg_frames_per_lu = tot_frames_per_lu * 1.0 / len(lu_to_frame_dict)

    sys.stderr.write("# Max frames for LU: %d in LU (%s)\n"
                     "# Avg LUs for frame: %f\n"
                     "# Avg frames per LU: %f\n"
                     "# Max LUs for frame: %d in Frame (%s)\n"
                     % (max_frames,
                        LUDICT.getstr(longestlu),
                        tot_lus/tot_frames,
                        avg_frames_per_lu,
                        max_lus,
                        FRAMEDICT.getstr(longestfrm)))

    return lu_to_frame_dict, related_lus


def get_wvec_map():
    if not os.path.exists(EMBEDDINGS_FILE):
        raise Exception("Pretrained embeddings file not found!", EMBEDDINGS_FILE)
    sys.stderr.write("\nReading pretrained embeddings from {} ...\n".format(EMBEDDINGS_FILE))
    if EMBEDDINGS_FILE.endswith('txt'):
        embs_file = EMBEDDINGS_FILE
    else:
        raise Exception('Pretrained embeddings file needs to be a text file, not archive!',
                        EMBEDDINGS_FILE)
    wvf = open(embs_file, 'r')
    wvf.readline()
    wd_vecs = {VOCDICT.addstr(line.split(' ')[0]):
                [float(f) for f in line.strip().split(' ')[1:]] for line in wvf}
    return wd_vecs


def get_chvec_map():
    if not os.path.exists(CHARACTER_EMBEDDINGS):
        raise Exception("Pretrained embeddings file not found!", CHARACTER_EMBEDDINGS)
    sys.stderr.write("\nReading pretrained embeddings from {} ...\n".format(CHARACTER_EMBEDDINGS))
    if CHARACTER_EMBEDDINGS.endswith('txt'):
        embs_file = CHARACTER_EMBEDDINGS
    else:
        raise Exception('Pretrained embeddings file needs to be a text file, not archive!',
                        CHARACTER_EMBEDDINGS)
    wvf = open(embs_file, 'r')
    wvf.readline()
    ch_vecs = {CHARDICT.addstr(line.split(' ')[0]):
                [float(f) for f in line.strip().split(' ')[1:]] for line in wvf}
    CHARDICT.lock()
    print(CHARDICT.size())
    ch_vecs_array = np.zeros((CHARDICT.size(), 300))
    char_indices = enumerate(list(CHARDICT._strtoint.keys()))
    for i, c in char_indices:
        idx = CHARDICT.addstr(c)
        if idx in ch_vecs.keys():
            ch_vec = ch_vecs[CHARDICT.addstr(c)]
            ch_vecs_array[i] = ch_vec
    #print(ch_vecs_array)
    pca = PCA(n_components=50)
    pca.fit(ch_vecs_array)
    ch_vecs_pca = np.array(pca.transform(ch_vecs_array))
    wvf = open(embs_file, 'r')
    wvf.readline()
    print([f for f in ch_vecs_pca[0]])
    ch_vecs = {CHARDICT.addstr(line.split(' ')[0]):[float(f) for f in ch_vecs_pca[CHARDICT.addstr(line.split(' ')[0])]] for line in wvf}

    return ch_vecs

def get_chains(node, inherit_map, path):
    if node in inherit_map:
        for par in inherit_map[node]:
            path = get_chains(par, inherit_map, path+[par])
    return path


def read_frame_relations():
    sys.stderr.write("\nReading inheritance relationships from {} ...\n".format(FRAME_REL_FILE))

    f = open(FRAME_REL_FILE, "rb")
    tree = et.parse(f)
    root = tree.getroot()

    relations = {}
    commonest_frame_child = None
    max_num_parents = 0
    paths = {}

    fe_relations = {}
    commonest_fe_child = None
    max_parent_fes = 0
    fepaths = {}

    for reltype in root.iter('{http://framenet.icsi.berkeley.edu}frameRelationType'):
        if reltype.attrib["name"] != "Inheritance":
            continue

        for relation in reltype.findall('{http://framenet.icsi.berkeley.edu}frameRelation'):
            sub_frame = FRAMEDICT.addstr(relation.attrib["subFrameName"])
            super_frame = FRAMEDICT.addstr(relation.attrib["superFrameName"])
            if sub_frame not in relations:
                relations[sub_frame] = []
            relations[sub_frame].append(super_frame)
            if len(relations[sub_frame]) > max_num_parents:
                max_num_parents = len(relations[sub_frame])
                commonest_frame_child = sub_frame

            for ferelation in relation.findall('{http://framenet.icsi.berkeley.edu}FERelation'):
                sub_fe = FEDICT.addstr(ferelation.attrib["subFEName"])
                super_fe = FEDICT.addstr(ferelation.attrib["superFEName"])
                if sub_fe != super_fe:
                    if sub_fe not in fe_relations:
                        fe_relations[sub_fe] = []
                    fe_relations[sub_fe].append(super_fe)
                    if len(fe_relations[sub_fe]) > max_parent_fes:
                        max_parent_fes = len(fe_relations[sub_fe])
                        commonest_fe_child = sub_fe

    f.close()

    for leaf in relations.keys():
        if leaf not in paths:
            paths[leaf] = []
        paths[leaf] += get_chains(leaf, relations, [])
    xpaths = {p:set(paths[p]) for p in paths}

    # TODO: not sure why there is a problem with getting the entire path for FE relations
    # TODO: for now, it's only one hop
    # for feleaf in fe_relations.keys():
    #     fe_relations[feleaf] = list(set(fe_relations[feleaf]))
    #     if feleaf not in fepaths:
    #         fepaths[feleaf] = []
    #     fepaths[feleaf] += get_chains(feleaf, fe_relations, [])
    # xfepaths = {p:set(fepaths[p]) for p in fepaths}

    sys.stderr.write("# descendant frames: %d commonest descendant = %s (%d parents)\n"
                     %(len(xpaths), FRAMEDICT.getstr(commonest_frame_child), max_num_parents))
    sys.stderr.write("# descendant FEs: %d commonest descendant = %s (%d parents)\n\n"
                     %(len(fe_relations), FEDICT.getstr(commonest_fe_child), max_parent_fes))

    return xpaths, fe_relations


def read_brackets(constitfile):
    sys.stderr.write("\nReading constituents from " + constitfile + " ...\n")
    reader = BracketParseCorpusReader(PARSER_DATA_DIR + "rnng/", constitfile)
    parses = reader.parsed_sents()
    return parses


def read_ptb():
    sys.stderr.write("\nReading PTB data from " + PTB_DATA_DIR + " ...\n")
    sentences = []
    senno = 0
    with codecs.open("ptb.sents", "w", "utf-8") as ptbsf:
        for constitfile in os.listdir(PTB_DATA_DIR):
            reader = BracketParseCorpusReader(PTB_DATA_DIR, constitfile)
            parses = reader.parsed_sents()
            # TODO: map from parses to sentences
            for p in parses:
                ptbsf.write(" ".join(p.leaves()) + "\n")
                tokpos = p.pos()
                tokens = [VOCDICT.addstr(tok) for tok,pos in tokpos]
                postags = [POSDICT.addstr(pos) for tok,pos in tokpos]
                s = Sentence("constit",sentnum=senno,tokens=tokens,postags=postags,)
                s.get_all_parts_of_ctree(p, CLABELDICT, False)
                sentences.append(s)
                senno += 1
        sys.stderr.write("# PTB sentences: %d\n" %len(sentences))
        ptbsf.close()
    return sentences

