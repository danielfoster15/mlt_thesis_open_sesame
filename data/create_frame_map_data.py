import pandas as pd
import os
# with open('lu_fixed.txt', 'w') as f:
#     f1 = open('lus.txt')
#     for line in f1:
#         line_items = line.strip().split()
#         if len(line_items) == 3:
#             lu = line_items[0]+'.'+line_items[1]
#             frame = line_items[2]
#             f.write(lu+'\t'+frame+'\n')
#         elif len(line_items) == 2:
#             lu = line_items[0]+'.'+'n'
#             frame = line_items[1]
#             f.write(lu+'\t'+frame+'\n')
#         elif len(line_items) == 0:
#             continue

lu_df = pd.read_csv('lu_fixed.txt', sep='\t', header=None, index_col=None, names=['lu', 'frame'])
fe_df = pd.read_csv('frame_elements.txt', sep='\t', header=None, index_col=None, names=['frame', 'fe'])

frame_df = pd.merge(lu_df, fe_df, how='outer', on='frame')



frames = frame_df['frame'].unique().tolist()
print(frames)
print(len(frames))

for frame in frames:
    f = open('fndata-1.7/frame/'+frame+'.txt', 'w')
    lus = frame_df[frame_df['frame'] == frame]['lu'].dropna().unique().tolist()
    fes = frame_df[frame_df['frame'] == frame]['fe'].dropna().unique().tolist()
    if len(lus) > 0:
        f.write('Frame' +'\t' + frame+'\n')
        f.write('LUs'+'\t')
        for lu in lus:
            f.write(lu+'\t')
        f.write('\n')
        f.write('FEs'+'\t')
        for fe in fes:
            f.write(fe+'\t')
        f.write('\n')
        f.close()
    else:
        f.close()
        os.remove('fndata-1.7/frame/'+frame+'.txt')