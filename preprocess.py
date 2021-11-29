"""Preprocess Template.

This script will be invoked in two ways during the Unearthed scoring pipeline:
 - first during model training on the 'public' dataset
 - secondly during generation of predictions on the 'private' dataset
"""
import argparse
import logging
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Add Feature: holidays
# From https://data.gov.au/data/dataset/australian-holidays-machine-readable-dataset and https://www.michaelplazzer.com/a-better-public-holiday-data-set/
# Add Feature: holidays
# From https://data.gov.au/data/dataset/australian-holidays-machine-readable-dataset and https://www.michaelplazzer.com/a-better-public-holiday-data-set/
wa_hols = [
    '2009-01-01', '2009-01-26', '2009-03-02', '2009-04-10', '2009-04-11',
    '2009-04-12', '2009-04-13', '2009-04-25', '2009-04-27', '2009-06-01',
    '2009-09-28', '2009-12-25', '2009-12-26', '2009-12-28', '2010-01-01',
    '2010-01-26', '2010-03-01', '2010-04-02', '2010-04-05', '2010-04-26',
    '2010-06-07', '2010-09-27', '2010-12-25', '2010-12-26', '2010-12-27',
    '2010-12-28', '2011-01-01', '2011-01-26', '2011-03-07', '2011-04-22',
    '2011-04-25', '2011-04-26', '2011-06-06', '2011-10-28', '2011-12-25',
    '2011-12-26', '2011-12-27', '2012-01-01', '2012-01-02', '2012-01-26',
    '2012-03-05', '2012-04-06', '2012-04-09', '2012-04-25', '2012-06-04',
    '2012-10-01', '2012-12-25', '2012-12-26', '2013-01-01', '2013-01-26',
    '2013-03-04', '2013-03-29', '2013-04-01', '2013-04-25', '2013-06-03',
    '2013-09-30', '2013-12-25', '2013-12-26', '2014-01-01', '2014-01-27',
    '2014-03-03', '2014-04-18', '2014-04-19', '2014-04-21', '2014-04-25',
    '2014-06-02', '2014-09-29', '2014-12-25', '2014-12-26', '2015-01-01',
    '2015-01-26', '2015-03-02', '2015-04-03', '2015-04-04', '2015-04-06',
    '2015-04-25', '2015-04-27', '2015-06-01', '2015-09-28', '2015-12-25',
    '2016-01-01', '2016-01-26', '2016-03-07', '2016-03-25', '2016-03-28',
    '2016-04-25', '2016-06-06', '2016-09-26', '2016-12-25', '2016-12-26',
    '2016-12-27', '2017-01-01', '2017-01-02', '2017-01-26', '2017-03-06',
    '2017-04-14', '2017-04-17', '2017-04-25', '2017-06-05', '2017-09-25',
    '2017-12-25', '2017-12-26', '2018-01-01', '2018-01-26', '2018-03-05',
    '2018-03-30', '2018-04-02', '2018-04-25', '2018-06-04', '2018-09-24',
    '2018-12-25', '2018-12-26', '2019-01-01', '2019-01-28', '2019-03-04',
    '2019-04-19', '2019-04-22', '2019-04-25', '2019-06-03', '2019-09-30',
    '2019-12-25', '2019-12-26', '2020-01-01', '2020-01-27', '2020-03-02',
    '2020-04-10', '2020-04-13', '2020-04-25', '2020-04-27', '2020-06-01',
    '2020-09-28', '2020-12-25', '2020-12-26', '2020-12-28', '2021-01-01',
    '2021-01-26', '2021-03-01', '2021-04-02', '2021-04-05', '2021-04-25',
    '2021-04-26', '2021-06-07', '2021-09-27', '2021-12-25', '2021-12-26',
    '2021-12-27', '2021-12-28'
]
wa_hols = pd.to_datetime(wa_hols, format="%Y-%m-%d")

# We have thousands of work descriptions, lets just use the top 50, the rest will be "Other"
WORK_DESC_TOP = [
    "Training",
    "Admin",
    "TCS: Pole Distribution",
    "NC- Distribution Standard Jobs",
    "TCS: Conductor",
    "Safety",
    "SEQT - Payroll Costed Hours",
    "Housekeeping",
    "TCS: NP   No Power",
    "Data Maintenence",
    "CUSTOMER SERVICE - PAYROLL COSTED HOURS",
    "Meeting/briefings",
    "NC- Transmission Standard Jobs",
    "Commercial - Payroll Costed Hours",
    "Downtime",
    "CONSTRUCTION",
    "Visual Management",
    "F&M - Payroll Costed Hours",
    "REGION NORTH - L3 STORM DAMAGE",
    "Ops Maintenance - Payroll Costed Hours",
    "TCS: PB   Pole Broken/Damaged",
    "Data Management",
    "Property & Fleet - Payroll Costed Hours",
    "NMP&D - Payroll Costed Hours",
    "BP&R - Payroll Costed Hours",
    "NCIMP Continuous Improvement",
    "TCS: Distribution Transformer",
    "TCS: RT   Recloser Trip",
    "REGION METRO - L3 STORM DAMAGE",
    "Ten Hour Break",
    "TCS: LV Cable Underground",
    "Temporary Disconnection Only",
    "TCS: Pole Distribution (T-Fixed)",
    "General Training & Travel",
    "HUMAN RESOURCES - PAYROLL COSTED HOURS",
    "Fault Ready",
    "TCS: HV Cable Underground",
    "TCS: ES   Electric Shock",
    "TCS: DOFT Drop Out Fuse Trip",
    "Project Scoping",
    "TCS: LV Cross Arm",
    "Network Ops - Payroll Costed Hours",
    "TCS: PP   Part Power",
    "TCS: MC   Miscellaneous Non Hazard",
    "CCS PAYROLL COSTED HOURS",
    "Project Management",
    "METER REPLACEMENT - Internal",
    "TCS: MH   Miscellaneous Hazard",
    "DFIS/DFMS Data Correction Metro",
    "Level 3 Event Waroona B/fires Jan 2016",
]

WORK_DESC_KEY = [
                 'TCS:', 
                 'POLE',
                 'DESIGN',
                 'PROJECT', 
                 ]


input_cols = [
    "TIME_TYPE",
    "FUNC_CAT",
    "TOT_BRK_TM",
    "hour",
    "holiday",
#     "day_of_year",
#     "week"
]
#input_cols = [
 #   "WORK_DESC_top",
 #   "TIME_TYPE",
 #   "FUNC_CAT",
 #   "TOT_BRK_TM",
 #   "hour",
 #   "holiday",
 #   "day_of_year",
 #   "day_name",
 #   "week",
#]
for i in WORK_DESC_KEY:
    input_cols.append(i)

def top_work_desc_feat(x):
    return x if x in WORK_DESC_TOP else "Other"


def shuffle_df(df, random_seed=42):
    return df.sample(frac=1, random_state=random_seed, replace=False)


def oversample(df, y_col, n=None, random_state=42):
    """Sample an equal amount from each class, with replacement"""
    gs = [g for _, g in df.groupby(y_col)]
    if n is None:
        n = max(len(g) for g in gs)

    # sample equal number of each group
    gs = [g.sample(n, random_state=random_state, replace=True) for g in gs]
    # concat, and shuffle
    df = pd.concat(gs, 0)
    df = shuffle_df(df)
    return df


def preprocess(data_file, is_training = True):
    """Apply preprocessing and featurization steps to each file in the data directory.

    Your preprocessing and feature generation goes here.

    We've added some basic temporal features as an example.
    """
    logger.info(f"running preprocess on {data_file}")

    # read the data file
    df = pd.read_csv(
        data_file,
        parse_dates=[
            0,
        ],
    )
    logger.info(f"data read from {data_file} has shape of {df.shape}")

    df["WORK_DESC_top"] = df["WORK_DESC"].apply(top_work_desc_feat)
    
    import numpy as np
    df_key = pd.DataFrame(np.zeros((len(df), len(WORK_DESC_KEY))),columns=WORK_DESC_KEY)
    df_work_desc = df['WORK_DESC'][:]
    for t in range (0,len(df)):
        c = df_work_desc[t].split(" ")
        for tt in WORK_DESC_KEY:
            for cc in c:
                if cc == tt:
                    df_key.iloc[t][tt]=1   
    for tt in WORK_DESC_KEY:
        df[tt]=df_key[tt]

    dt = df.Work_DateTime.dt
    df["hour"] = dt.hour
    df["day_of_week"] = dt.dayofweek
    df["day_of_year"] = dt.dayofyear
    #df["day_name"] = dt.day_name()
    df["week"] = dt.week

    # Was the day a Western Australia public holiday?
    df["holiday"] = dt.round("1D").isin(wa_hols)
    
    df.fillna(0,inplace=True)
    
    #df["Incident_Number"]=df["Incident_Number"].apply(lambda x: 1 if x!=0 else 0)

    logger.info(f"data after preprocessing has shape of {df.shape}")
    if is_training:
        return df[input_cols + ['incident']]
    else:
        df = pd.get_dummies(df[input_cols]) #delete
        df.drop(columns='TIME_TYPE_Overtime', inplace = True)
        return df


if __name__ == "__main__":
    """Preprocess Main

    The main function is called by both Unearthed's SageMaker pipeline and the
    Unearthed CLI's "unearthed preprocess" command.

    WARNING - modifying this file may cause the submission process to fail.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, default="/opt/ml/processing/input/public/public.csv.gz"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/opt/ml/processing/output/preprocess/public.csv.gz",
    )
    args, _ = parser.parse_known_args()

    # call preprocessing on private data
    df = preprocess(args.input, False)

    # remove the target columns
    # the target variables are labelled target_pressure_absolute_[1dy, 2wk, 6wk]
    target_columns = ['WORK_NO']
    try:
        df.drop(columns=target_columns, inplace=True)
    except KeyError:
        pass
    logger.info(f"preprocessed result shape is {df.shape}")

    # write to the output location
    df.to_csv(args.output)
