def drop_constant_columns(df):
    # Lista para almacenar las columnas que serán eliminadas
    columns_dropped = []
    
    for column in df.columns:
        # Intentar calcular nunique(), ignorando las columnas que contienen tipos no hasheables
        try:
            if df[column].nunique() == 1:
                df.drop(columns=[column], inplace=True)
                columns_dropped.append(column)
        except TypeError:
            print(f"Column '{column}' was skipped due to unhashable data types.")
    
    if columns_dropped:
        print("The following columns were dropped because all values were the same:", columns_dropped)
    else:
        print("No columns were dropped. All columns have multiple unique values or unhashable types.")
