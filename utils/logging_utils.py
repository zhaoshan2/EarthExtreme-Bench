import pandas as pd
def save_errorScores(csv_path, value, error_type):
    score = pd.DataFrame.from_dict(value,
                                orient='index')
    score.reset_index(inplace=True)
    column_means = score.iloc[:, 1:].mean()

    # Append the means to the dataframe as the last row
    # Set the 'index' column of the new row to a specific identifier
    means_row = pd.DataFrame([['mean'] + column_means.tolist()], columns=score.columns)
    score = pd.concat([score, means_row], ignore_index=True)
    # score.loc[len(score.index)] = ['avg', score.iloc[:, 1:].mean().tolist()]#sum(value.values()) / len(value)]
    # Write DataFrame to CSV
    score.to_csv("{}/{}.csv".format(csv_path, f'{error_type}'), index=False)