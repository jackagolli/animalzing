import pandas as pd
from pathlib import Path
import itertools
from collections import Counter
import numpy as np


# 2 X 0 = 1
# 1 x  = 0.5 1, 0.5 0
# 0 x 0 = 0

def mapGenotype(flower_str):
    flower_genotype = []
    flower = list(flower_str)
    for i in range(len(flower)):
        if i == 0:
            if flower[i] == '0':
                flower_genotype.append('rr')
            elif flower[i] == '1':
                flower_genotype.append('Rr')
            elif flower[i] == '2':
                flower_genotype.append('RR')
        elif i == 1:
            if flower[i] == '0':
                flower_genotype.append('yy')
            elif flower[i] == '1':
                flower_genotype.append('Yy')
            elif flower[i] == '2':
                flower_genotype.append('YY')
        elif i == 2:
            if flower[i] == '0':
                flower_genotype.append('ww')
            elif flower[i] == '1':
                flower_genotype.append('Ww')
            elif flower[i] == '2':
                flower_genotype.append('WW')
        # i==3 is for roses only

    # flower_genotype = ''.join(flower_genotype)

    return flower_genotype


def permute(str1, str2, n):
    combinations = [] * n

    for i in range(n):
        combinations.append([''.join(sorted(p)) for p in itertools.product(str1[i], str2[i])])

    combinations = list(map(''.join, zip(*combinations)))
    combinations_df = pd.DataFrame(combinations)
    normalized = combinations_df.value_counts(normalize=True)

    return normalized


def determineColors(genotype, table):
    colors = []
    for i, j in genotype.items():
        single_genotype = i[0]
        colors.append(table.loc[table['genotype'] == single_genotype]['color'].values[0])

    return colors


def process(genotype_series, colors_list):

    df = pd.DataFrame(genotype_series, columns=['probability'])
    df['color'] = colors_list
    df = df.reset_index()
    df = df.rename({0: 'genotype'}, axis='columns')
    df = df.reindex(columns=['genotype','color','probability'])

    return df



flower_1 = '112'
flower_2 = '211'

n = len(flower_1)
flower_type = 'cosmos'
lookup_table = pd.read_csv(f'data/{flower_type}.csv', names=['genotype', 'color'])

flower_1_genotype = mapGenotype(flower_1)
flower_2_genotype = mapGenotype(flower_2)

combined_genotype = permute(flower_2_genotype, flower_1_genotype, n)
colors = determineColors(combined_genotype, lookup_table)
genotype_df = process(combined_genotype,colors)

print(f'flower 1 ({flower_1}) + flower 2 ({flower_2}) = ')
print(genotype_df)
