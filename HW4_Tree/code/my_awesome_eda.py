import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

COLUMN_TYPES = {
    0: 'Number',
    1: 'String',
    2: 'Factor'
}

columns_names_sorted = {
    COLUMN_TYPES[0]: [],
    COLUMN_TYPES[1]: [],
    COLUMN_TYPES[2]: [],
}

def run_eda(df):
    for col_name in df.columns:
        col_wo_na = df[col_name].dropna()
        val_count = col_wo_na.value_counts()

        is_factor = False
        if len(val_count.keys()) * 100 / len(col_wo_na) < 1:
            avg_freq = 0
            for factor_value in val_count.keys():
                avg_freq += val_count[factor_value] / len(col_wo_na)
            avg_freq = avg_freq / len(val_count.keys())
            is_factor = avg_freq > 0.1
            if is_factor:
                columns_names_sorted[COLUMN_TYPES[2]].append(col_name)
        if not is_factor:
            if pd.api.types.is_numeric_dtype(col_wo_na.dtype):
                columns_names_sorted[COLUMN_TYPES[0]].append(col_name)
            if pd.api.types.is_string_dtype(col_wo_na.dtype):
                columns_names_sorted[COLUMN_TYPES[1]].append(col_name)

    print()
    print()

    print('------------------------------ WELCOME TO EDA! --------------------'
          '----------')
    print('------------------------------ mmm... -----------------------------'
          '-')
    print('------------------------------ your data is so delicious... -------'
          '-----------------------')
    print('''
                    (
           (      )     )
             )   (    (
            (          `
        .-""^"""^""^"""^""-.
      (//\\\\//\\\\//\\\\//\\\\//\\\\//)
       ~\^^^^^^^^^^^^^^^^^^/~
         `================`
    ''')

    print('------------------------------ WELCOME TO EDA! --------------------'
          '----------')

    print()
    print()

    print(f'--------------------- Found \'{COLUMN_TYPES[2]}\' columns --------'
          f'------------- ')
    for factor_col_name in columns_names_sorted[COLUMN_TYPES[2]]:
        col_wo_na = df[factor_col_name].dropna()
        val_count = col_wo_na.value_counts()
        print(f'\'{factor_col_name}\' column')
        print(f'Counts of factors: {len(val_count.keys())}')
        for factor_value in val_count.keys():
            freq = round(val_count[factor_value] / len(col_wo_na), 2)
            print(f'{factor_value} - '
                  f'count: {val_count[factor_value]}; '
                  f'frequency: {freq}')
        print()

    print()

    print(f'--------------------- Found \'{COLUMN_TYPES[0]}\' columns --------'
          f'------------- ')
    for num_col_name in columns_names_sorted[COLUMN_TYPES[0]]:
        col_wo_na = df[num_col_name].dropna()
        print(f'\'{num_col_name}\' column')
        print(f'Min: {col_wo_na.min()}')
        print(f'Max: {col_wo_na.max()}')
        print(f'Mean: {round(col_wo_na.mean(), 2)}')
        print(f'Std: {round(col_wo_na.std(), 2)}')
        print(f'q0.25: {round(col_wo_na.quantile(0.25), 2)}')
        print(f'Median: {round(col_wo_na.median(), 2)}')
        print(f'q0.75: {round(col_wo_na.quantile(0.75), 2)}')
        print()

    print()

    print(f'--------------------- Quantity of ejection values ----------------'
          f'----- ')
    for num_col_name in columns_names_sorted[COLUMN_TYPES[0]]:
        count_not_ejection = 0
        col_wo_na = df[num_col_name].dropna()
        Q1 = col_wo_na.quantile(0.25)
        Q3 = col_wo_na.quantile(0.75)
        IQR = Q3 - Q1
        for i in range(0, len(col_wo_na)):
            if col_wo_na.iloc[i] >= Q1 - 1.5 * IQR \
                    and col_wo_na.iloc[i] <= Q3 + 1.5 * IQR:
                count_not_ejection += 1
        print(f'{num_col_name} count of ejection: '
              f'{len(col_wo_na) - count_not_ejection}')

    print()

    print(f'--------------------- N/A Summary --------------------- ')
    columns_with_na = df.isna().sum()
    print(f'Count of all N/A in DF: {columns_with_na.sum()}')
    print(f'Count of rows that contain N/A: {len(df[df.isna().any(axis=1)])}')
    print('Column names that contain N/A:')
    for column_name_contains_na in columns_with_na.keys():
        if columns_with_na[column_name_contains_na] > 0:
            print(f'{column_name_contains_na}')

    all_na_count = {
        'ColumnName': [],
        'NaCount': [],
        'Share': [],
        'Number': []
    }
    row_number = 0
    for col_name in df.columns:
        all_na_count['ColumnName'].append(col_name)
        all_na_count['NaCount'].append(df[col_name].isna().sum())
        all_na_count['Share'].append(df[col_name].isna().sum() / len(df[col_name]))
        all_na_count['Number'].append(row_number)
        row_number += 1
    na_count_df = pd.DataFrame(all_na_count)


    print(na_count_df)

    sns.set_style('whitegrid')
    fig_nas_plot, ax_nas_plot = plt.subplots()
    sns.barplot(data=na_count_df, x='ColumnName', y='Share', color='red', ax=ax_nas_plot)
    ax_nas_plot.set_title('Share of N/A for each column')
    plt.xticks(rotation=45, ha='right')
    fig_nas_plot.show()

    print()



    print(f'--------------------- Duplicates in DF --------------------- ')
    duplicates = df[df.duplicated()]
    print(f'Quantity of duplicates in df: {len(duplicates)}')




    df_number = df[columns_names_sorted[COLUMN_TYPES[0]]]

    correclation_mtx = df_number.corr()
    print(correclation_mtx)
    fig_heatmap, ax_heatmap = plt.subplots()
    ax_heatmap.set_title('Correlation matrix of numeric column values')
    sns.heatmap(correclation_mtx, cmap="YlGnBu", ax=ax_heatmap)


    print(df_number)

    ncols = 2
    nrows = int(len(columns_names_sorted[COLUMN_TYPES[0]]) / ncols)
    if nrows <= 0:
        nrows = 1
    if nrows * ncols < len(columns_names_sorted[COLUMN_TYPES[0]]):
        nrows += 1


    fig, ax = plt.subplots(nrows, ncols, figsize=(20, 10))

    next_col_name_idx = 0
    for ax_row in range(0, nrows):
        if next_col_name_idx >= len(columns_names_sorted[COLUMN_TYPES[0]]):
            break
        for ax_col in range(0, ncols):
            if next_col_name_idx >= len(columns_names_sorted[COLUMN_TYPES[0]]):
                break
            col_name = columns_names_sorted[COLUMN_TYPES[0]][next_col_name_idx]
            ax[ax_row, ax_col].hist(df_number[col_name], bins=100)
            ax[ax_row, ax_col].set_title(col_name)
            next_col_name_idx += 1



