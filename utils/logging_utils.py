import pandas as pd
def save_errorScores(csv_path, value, error_type):
    score = pd.DataFrame.from_dict(value,
                                orient='index',
                                columns=['Score'])
    score.reset_index(inplace=True)
    score.columns = ['Target time', 'Score']
    score.loc[len(score.index)] = ['avg', sum(value.values()) / len(value)]
    # Write DataFrame to CSV
    score.to_csv("{}/{}.csv".format(csv_path, f'{error_type}'), index=False)