import pandas as pd
import numpy as np
import itertools
from datetime import date

# Converts 3 digit number (XXX) string to actual genotype
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

    return flower_genotype

# finds all possible unique combinations of genes (Punnett square)
def permute(str1, str2, n):
    combinations = [] * n

    for i in range(n):
        combinations.append([''.join(sorted(p)) for p in itertools.product(str1[i], str2[i])])

    combinations = list(map(''.join, zip(*combinations)))
    combinations_df = pd.DataFrame(combinations)
    normalized = combinations_df.value_counts(normalize=True)

    return normalized

# Lookup color based on given genotype
def determineSingleColor(genotype, table):
    color = table.loc[table['genotype'] == genotype]['color'].values[0]

    return color

# Determines related colors to probability dataframe of genotypes
def determineColors(genotype, table):
    colors = []
    for i, j in genotype.items():
        single_genotype = i[0]
        colors.append(table.loc[table['genotype'] == single_genotype]['color'].values[0])

    return colors

# Formats dataframe to look nice
def process(genotype_series, colors_list):
    df = pd.DataFrame(genotype_series, columns=['probability'])
    df['color'] = colors_list
    df = df.reset_index()
    df = df.rename({0: 'genotype'}, axis='columns')
    df = df.reindex(columns=['genotype', 'color', 'probability'])

    return df

# Returns possible genotypes for a given color
def returnCombinations(color, lookup_table):
    filtered_df = lookup_table.loc[lookup_table['color'] == color]

    return filtered_df


# Select 2 flower types here to see possible hybrids
def determineProbability(flower_1, flower_2, lookup_table):
    n = len(flower_1)

    flower_1_genotype = mapGenotype(flower_1)
    flower_2_genotype = mapGenotype(flower_2)
    combined_genotype = permute(flower_2_genotype, flower_1_genotype, n)
    colors = determineColors(combined_genotype, lookup_table)
    genotype_df = process(combined_genotype, colors)

    return genotype_df

# See all possible flower combinations to produce a given color hybrid
def reverseLookupColor(color, lookup_table):

    all_genotypes = [''.join(p) for p in itertools.product('012', repeat=3)]
    all_genotypes_copy = all_genotypes

    combinations_array = np.zeros((1, 2))

    for genotype1 in all_genotypes:

        for genotype2 in all_genotypes_copy:
            combinations_array = np.append(combinations_array, [[genotype1, genotype2]], axis=0)

    combinations_array = np.delete(combinations_array, 0, axis=0)
    combinations_array = np.insert(combinations_array, 1,
                                   np.zeros((combinations_array.shape[0])),
                                   axis=1)
    combinations_array = np.append(combinations_array,
                                   np.zeros((combinations_array.shape[0], 1)),
                                   axis=1)

    i = 0
    for x in combinations_array:

        flower1 = x[0]
        flower2 = x[2]
        flower1_g = ''.join(mapGenotype(flower1))
        flower2_g = ''.join(mapGenotype(flower2))
        f1_color = determineSingleColor(flower1_g, lookup_table)
        f2_color = determineSingleColor(flower2_g, lookup_table)

        combinations_array[i, 1] = f1_color
        combinations_array[i, 3] = f2_color

        hybrid = determineProbability(flower1, flower2, lookup_table)
        matches = hybrid.loc[hybrid['color'] == color]
        additional_cols = len(matches) * 2
        comb_arr_size = combinations_array.shape

        if 4 + additional_cols > comb_arr_size[1]:
            arr = np.zeros((comb_arr_size[0], (additional_cols + 4) - comb_arr_size[1]))
            combinations_array = np.append(combinations_array, arr, axis=1)

        newlist = []
        if not matches.empty:
            for j, row in matches.iterrows():
                newlist.append(row['genotype'])
                newlist.append(row['probability'])

            combinations_array[i, 4:4 + len(newlist)] = newlist
            i += 1
        else:
            combinations_array = np.delete(combinations_array, i, axis=0)

    results_df = pd.DataFrame(combinations_array)
    results_df = results_df.rename(columns={0: 'flower1', 1: 'color1', 2: 'flower2', 3: 'color2'})

    return results_df

# Enter flower type
flower_type = 'cosmos'
lookup_table = pd.read_csv(f'data/{flower_type}.csv', names=['genotype', 'color'])

# 1) Lookup a flower color
# flower = '120'
# genotype = ''.join(mapGenotype(flower))
# print(determineSingleColor(genotype,lookup_table))

# 2) Input two flowers to see possible hybrids and probabilities
# flower_1 = '110'
# flower_2 = '210'
# flower_type = 'cosmos'
# print(f'flower 1 ({flower_1}) + flower 2 ({flower_2}) = ')
# print(determineProbability(flower_1,flower_2,lookup_table))

# 3) Select a color to see possible parents
color = 'Pink'
flower_type = 'cosmos'
results = reverseLookupColor(color, lookup_table)
print(results)
today = date.today()
results.to_csv(f'{today}.csv',index=False)

