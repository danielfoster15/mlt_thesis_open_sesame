import pandas as pd
lu_df = pd.read_csv('lus.txt', sep='\t', header=None, index_col=None, names=['lu', 'lupos', 'frame'])
fe_df = pd.read_csv('frame_elements.txt', sep='\t', header=None, index_col=None, names=['frame', 'fe'])

frame_df = pd.merge(lu_df, fe_df, how='outer', on='frame')



frames = frame_df['frame'].unique().tolist()
print(frames)
print(len(frames))

for frame in frames:
    with open('fndata-1.7/frame/'+frame+'.txt', 'w') as f:
        lus = frame_df[frame_df['frame'] == frame]['lu'].dropna().unique().tolist()
        fes = frame_df[frame_df['frame'] == frame]['fe'].dropna().unique().tolist()
        if len(lus) > 0:
            f.write('LUs'+'\t')
            for lu in lus:
                f.write(lu+'\t')
            f.write('\n')
            f.write('FEs'+'\t')
            for fe in fes:
                f.write(fe+'\t')
            f.write('\n')