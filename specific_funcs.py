import pandas as pd
import numpy as np
import re
import os

from funcs import character_rules


PROJECT_PATH = os.getcwd()
DATA_PATH = os.path.join(PROJECT_PATH, 'data')

tyres_data = pd.read_csv(DATA_PATH+'\Tyre_sizes.csv', sep='\t', index_col=0)

def inch_to_mm(f):
    return f * 25.4

def apply_function_elementwise(values, func):
    for val in values:
        yield func(val)

def unify_pistons(s):
    s = s.lower()
    new = None
    if re.findall(r'([0-9]+)\s+(pistons?)\b',s):
        new = re.sub(r'([0-9]+)\s+(pistons?)\b', '\\1piston',s)
    elif re.findall(r'([0-9]+)-(pistons?)\b',s):
        new = re.sub(r'([0-9]+)-(pistons?)\b', '\\1piston',s)
    numkeys = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    nums = {k:v+1 for v,k in enumerate(numkeys)}
    nums['twin'] = 2
    for numstring, num in nums.items():
        """TODO - Try this, not yet tested"""
        pattern1 = r'(' + numstring + r')\s+(pistons?)\b'
        pattern2 = r'(' + numstring + r')-(pistons?)\b'
        pattern = None
        if re.findall(pattern1,s):
            pattern = pattern1
        elif re.findall(pattern2,s):
            pattern = pattern2
        substitution_1, substitution_2 = r'(' + numstring + r')', str(num)
        if pattern:
            new = re.sub(pattern, '\\1piston',s)
            new = re.sub(substitution_1, substitution_2, new)
    return new if new else s

########## TYRES FUNCTIONS ##########
def clean_tyre_description(s):
    s = s.upper()
    new_s = ''
    for c in s:
        if character_rules(c):
            new_s += c
    return new_s

def tyre_speed_and_construction(keys='upper'):
    labels = tyres_data.index.tolist()
    for i, label in enumerate(labels):
        if label == 'ZR':
            labels[i] = 'Z'
    speeds = tyres_data.iloc[:,0].tolist()
    if keys == 'lower':
        labels = [label.lower() for label in labels]
    return {k:v for (k,v) in zip(labels, speeds)}

def parse_standard_tyre(s):
    width, ratio, speed_construction, diameter = re.findall(r'([0-9]+)/([0-9]+)\-([A-Z]+)?([0-9]+)',s)[0]
    speed_dict = tyre_speed_and_construction()
    if len(speed_construction) > 0:
        if len(speed_construction) > 1:
            #speed = speed_dict[speed_construction]
            speed = speed_construction[0] if speed_construction[0] in speed_dict.keys() and speed_construction[0] not in ['B','R'] else 'A'
            construction = speed_construction[1] if speed_construction[1] in ['B','R'] else 'B'
            if speed == 'A' and construction == 'B':
                speed = speed_construction[1] if speed_construction[1] in speed_dict.keys() and speed_construction[1] not in ['B','R'] else 'A'
                construction = speed_construction[0] if  speed_construction[0] in ['B','R'] else 'B'
        else:
            letter = speed_construction
            if letter in speed_dict.keys() and letter not in ['B','R']: # This means the letter refers to speed
                speed = letter
                construction = 'B' # Construction is B by default
            elif letter in ['R', 'B']:
                construction = letter
                speed = 'A' # Equivalent to not indicated
            else:   # Found another letter that does not belong to any known classification
                construction = 'B'
                speed = 'A'
    else:
        construction, speed = 'B', 'A'

    return width, ratio, speed, construction, diameter

def parse_numeric_tyre(s):
    width, diameter = re.findall(r'(\d*[\.]?\d*)-(\d+)', s)[0]
    width, diameter = apply_function_elementwise([width, diameter], float)
    height = np.nan
    construction = 'B'
    #speed = np.nan
    return width, height, construction, diameter

def parse_alphanumeric_tyre(s):
    alphas = ['H', 'J', 'M', 'N', 'P', 'R', 'T', 'U', 'V']
    nums = [80,90,100,110,110,120,130,140,150]
    alphanum_width = {k:v for (k,v) in zip(alphas, nums)}
    width, ratio, construction, diameter  = re.findall(r'(M[A-Z]+)(\d+)-([A-Z])(\d+)', s)[0]
    width = (alphanum_width[width[1]])
    height = width * float(ratio) / 100
    diameter = inch_to_mm(float(diameter))
    return width, height, construction, diameter

def get_tyre_data(s):
    # International standard
    try:
        width, ratio, speed, construction, diameter = parse_standard_tyre(s)            
        width, ratio, diameter = apply_function_elementwise([width, ratio, diameter], float)
        height = width * ratio /100
        diameter = inch_to_mm(diameter) if diameter < 100 else diameter
        label_format = 'I'
    except:
        try:    #Numeric
            width, height, construction, diameter = parse_numeric_tyre(s)
            # Convert to inches except when the value is so big it must be already in mm
            width = inch_to_mm(width) if width < 10 else width
            diameter = inch_to_mm(diameter) if diameter < 50 else diameter
            speed = 'A'
            label_format = 'N'
        except:
            try:
                width, height, construction, diameter = parse_alphanumeric_tyre(s)
                diameter = inch_to_mm(diameter) if diameter < 50 else diameter
                speed = 'A'
                label_format = 'A'
            except IndexError:
                #print('Does not correspond to any known format')
                width, height, speed, construction, diameter, label_format = np.full((6,), np.nan)
    return pd.Series([width, height, speed, construction, diameter, label_format])

def tyres_columns(df, col_name):
    df[col_name] = df.loc[:,col_name].apply(clean_tyre_description)
    new_columns, tyre_properties = [], ['width', 'height', 'speed', 'construction', 'diameter', 'label_format']
    for property in tyre_properties:
        new_columns.append(col_name + '_' + property)
    df[new_columns] = df[col_name].apply(get_tyre_data)
    return df

def fill_tyre_width(df, col_name):
    unique_failed_classes = df[df[col_name].isnull()]['Category'].unique()
    for category in unique_failed_classes:
        avg = df.loc[df['Category']==category, col_name].mean()
        df.loc[df['Category']==category,col_name] =\
            df.loc[df['Category']==category, col_name].fillna(value=avg)
    return df

def fill_tyre_height(df, col_name):
    width_column = col_name.replace('height', 'width')
    unique_failed_classes = df[df[col_name].isnull()]['Category'].unique()
    for category in unique_failed_classes:
        cat_filter = df['Category']==category
        avg_height = df.loc[cat_filter, col_name].mean()
        avg_width = df.loc[cat_filter, width_column].mean()
        avg_ratio = avg_height / avg_width
        filler = df.loc[cat_filter, col_name].fillna(\
                df.loc[cat_filter, width_column] * avg_ratio)
        df.loc[cat_filter, col_name] = filler
    return df

def fill_tyre_diameter(df, col_name):
    unique_failed_classes = df[df[col_name].isnull()]['Category'].unique()
    for category in unique_failed_classes:
        avg = df.loc[df['Category']==category, col_name].mean()
        df.loc[df['Category']==category,col_name] =\
            df.loc[df['Category']==category, col_name].fillna(value=avg)
    return df

def fill_tyre_speed(df, col_name):
    avg = df.loc[:, col_name].mean()
    df.loc[:, col_name] = df.loc[:, col_name].fillna(value=avg)
    """unique_failed_classes = df[df[col_name].isnull()]['Category'].unique()
    for category in unique_failed_classes:
        avg = df.loc[df['Category']==category, col_name].mean()
        df.loc[df['Category']==category,col_name] =\
            df.loc[df['Category']==category, col_name].fillna(value=avg)
    """
    return df

#################################################