# -*- coding: utf-8 -*-
"""Remove exclusions from classifications

classification lists have text which include information on what is excluded from a category.

We wish to get rid of these exclusions so we can do text analysis on what is excluded

"""


def bracket_exclusions(df1,col_old):
    """ function to remove exclusions from the description which appear in brackets 
         takes in a dataframe and the target column, and returns the dataframe with the target column cleaned
    """
    df= df1.copy()
    ex_str = r'(\(except(.*?)\)|\(excl(.*?)\)|\(without(.*?)\)|\(not (.*?)\)|\(other than (.*?)\))'
    df['Excl_removed'] = ''
    df['Excl_removed'] = df[col_old].str.extract(ex_str)
    df[col_old+'_excl_rem'] = df[col_old].str.replace(ex_str, '')
    return df

def reg_excludes(df1,col_old, excl_col):
    """  function to remove phrases such as "except...", "excluding..." that appear at the end of the description
          takes in a dataframe with a column to be cleaned (col_old). 
          Returns the dataframe with col_old cleaned, and an extra column (excl_col) to contain the removed text
    """
    df= df1.copy()
    df['Temp'] = df[col_old]
    # remove exceptions in brackets
    ex_str = r'(\(except(.*?)\)|\(excl(.*?)\)|\(without(.*?)\)|\(not (.*?)\)|\(other than (.*?)\))'
    # unnecessary phrases:
    ex_str0 = r'(This(.*)excludes: |This(.*)excludes |This(.*)includes: |This(.*)includes )'
    # sometimes there is "include" info after a phrase of the pattern "except... ;" or "excelpt... ."
    ex_str1 = r'(, except.*?\.|, excl.*?\.|, without.*\.|, not\b.*?\.| not\b.*?\.| not\b.*?\)|, other than .*?\.)'
    ex_str1a = r'(, except.*?;|, excl.*?;|, without.*;|,\bnot\b.*?;|\bnot\b.*?;| not\b.*?\)|, other than .*?;)'

    # sometimes there is "include" info after a phrase of the pattern  "except..., and"
    ex_str1b = r'(, except.*?(, and)|, excl.*?(, and)|, without.*(, and))'
    # the phrases separated by a comma, eg "except..., " usually are followed by "exclude" info and occur at the end of the description.
    # however, phrases such as "not...", "other than..." which may have more "include" data later in the Description that must be left in.
    ex_str1c = r'(, whether or not.*?,)'
    # we now assume that the "except..." phrases continue to the end of the description field.
    ex_str2 = '(, except.*|, excl(.*)|, without .*| not\s.*|, other than .*)'    
    # other things to exclude
    ex_str3 = r'(not elsewhere classified | n.e.c. | n.e.c)'

    excl_list = []

    i = 1
    ex_strs = [ex_str, ex_str0, ex_str1, ex_str1a, ex_str1b, ex_str1c, ex_str2, ex_str3]
    for s in ex_strs:
        name = f'{excl_col}_{str(i)}'
        print(name)
        df[name] = ''
        df[name]= df['Temp'].str.extract(s, expand=True)
        df['Temp'] = df['Temp'].str.replace(s, '')
        
        df[excl_col] = df[excl_col].fillna('') + df[name].fillna('') 
        excl_list.append(name)

        i+=1
      #  get_string(df[df[name].notnull()][name])
    # create our new column with these items excluded
    df[col_old+'_excl_rem'] = df['Temp']
    # prepare dataframe to be returned

    df2 = df.copy().drop(excl_list +['Temp',col_old],axis=1)

    return df2