import sys
sys.path.append("../AIF360/")
import numpy as np
from tot_metrics import TPR, TNR
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools\
    import OptTools
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from aif360.datasets import StandardDataset
import warnings
import pandas as pd
warnings.simplefilter("ignore")
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

def get_distortion_compas(vold, vnew):
    """Distortion function for the compas dataset. We set the distortion
    metric here. See section 4.3 in supplementary material of
    http://papers.nips.cc/paper/6988-optimized-pre-processing-for-discrimination-prevention
    for an example

    Note:
        Users can use this as templates to create other distortion functions.

    Args:
        vold (dict) : {attr:value} with old values
        vnew (dict) : dictionary of the form {attr:value} with new values

    Returns:
        d (value) : distortion value
    """
    # Distortion cost
    distort = {}
    distort['two_year_recid'] = pd.DataFrame(
        {'No recid.': [0., 2.],
         'Did recid.': [2., 0.]},
        index=['No recid.', 'Did recid.'])
    distort['age_cat'] = pd.DataFrame(
        {'Less than 25': [0., 1., 2.],
         '25 to 45': [1., 0., 1.],
         'Greater than 45': [2., 1., 0.]},
        index=['Less than 25', '25 to 45', 'Greater than 45'])

    distort['c_charge_degree'] = pd.DataFrame(
        {'M': [0., 2.],
         'F': [1., 0.]},
        index=['M', 'F'])
    distort['priors_count'] = pd.DataFrame(
        {'0': [0., 1., 2., 100.],
         '1 to 3': [1., 0., 1., 100.],
         'More than 3': [2., 1., 0., 100.],
         'missing': [0., 0., 0., 1.]},
        index=['0', '1 to 3', 'More than 3', 'missing'])

    distort['score_text'] = pd.DataFrame(
        {'Low': [0., 2.],
         'MediumHigh': [2., 0.]},
        index=['Low', 'MediumHigh'])
    distort['sex'] = pd.DataFrame(
        {0.0: [0., 2.],
         1.0: [2., 0.]},
        index=[0.0, 1.0])
    distort['race'] = pd.DataFrame(
        {0.0: [0., 2.],
         1.0: [2., 0.]},
        index=[0.0, 1.0])

    total_cost = 0.0
    for k in vold:
        if k in vnew:
            total_cost += distort[k].loc[vnew[k], vold[k]]

    return total_cost


default_mappings = {
    'label_maps': [{1.0: 'Did recid.', 0.0: 'No recid.'}],
    'protected_attribute_maps': [{0.0: 'Male', 1.0: 'Female'},
                                 {1.0: 'Caucasian', 0.0: 'Not Caucasian'}]
}


def default_preprocessing(df):
    """Perform the same preprocessing as the original analysis:
    https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
    """
    return df[(df.days_b_screening_arrest <= 30)
              & (df.days_b_screening_arrest >= -30)
              & (df.is_recid != -1)
              & (df.c_charge_degree != 'O')
              & (df.score_text != 'N/A')]


class CompasDataset(StandardDataset):
    """ProPublica COMPAS Dataset.

    See :file:`aif360/data/raw/compas/README.md`.
    """

    def __init__(
            self,
            label_name='two_year_recid',
            favorable_classes=[0],
            protected_attribute_names=[
                'sex',
                'race'],
            privileged_classes=[
                ['Female'],
                ['Caucasian']],
            instance_weights_name=None,
            categorical_features=[
                'age_cat',
                'c_charge_degree',
                'c_charge_desc'],
            features_to_keep=[
                'sex',
                'age',
                'age_cat',
                'race',
                'juv_fel_count',
                'juv_misd_count',
                'juv_other_count',
                'priors_count',
                'c_charge_degree',
                'c_charge_desc',
                'two_year_recid',
                'length_of_stay'],
            features_to_drop=[],
            na_values=[],
            custom_preprocessing=default_preprocessing,
            metadata=default_mappings):

        

        def quantizePrior1(x):
            if x <= 0:
                return 0
            elif 1 <= x <= 3:
                return 1
            else:
                return 2

        def quantizeLOS(x):
            if x <= 7:
                return 0
            if 8 < x <= 93:
                return 1
            else:
                return 2

        def group_race(x):
            if x == "Caucasian":
                return 1.0
            else:
                return 0.0

        filepath = 'data/compas/compas-scores-two-years.csv'
        df = pd.read_csv(filepath, index_col='id', na_values=[])

        df['age_cat'] = df['age_cat'].replace('Greater than 45', 2)
        df['age_cat'] = df['age_cat'].replace('25 - 45', 1)
        df['age_cat'] = df['age_cat'].replace('Less than 25', 0)
        df['score_text'] = df['score_text'].replace('High', 1)
        df['score_text'] = df['score_text'].replace('Medium', 1)
        df['score_text'] = df['score_text'].replace('Low', 0)
        df['priors_count'] = df['priors_count'].apply(
            lambda x: quantizePrior1(x))
        df['length_of_stay'] = (pd.to_datetime(df['c_jail_out']) -
                                pd.to_datetime(df['c_jail_in'])).apply(
            lambda x: x.days)
        df['length_of_stay'] = df['length_of_stay'].apply(
            lambda x: quantizeLOS(x))
        df = df.loc[~df['race'].isin(
            ['Native American', 'Hispanic', 'Asian', 'Other']), :]
        df['c_charge_degree'] = df['c_charge_degree'].replace({'F': 0, 'M': 1})

        
        df['c_charge_degree'] = df['c_charge_degree'].replace({0: 'F', 1: 'M'})

        super(
            CompasDataset,
            self).__init__(
            df=df,
            label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop,
            na_values=na_values,
            custom_preprocessing=custom_preprocessing,
            metadata=metadata)

        
def reweight_df(dataset_orig_train):
    df_weight = dataset_orig_train.convert_to_dataframe()[0]
    df_weight['weight'] = 1
    df_weight['is_missing'] = 0
    df_weight['tmp'] = ''
    tmp_result = []
    for i, j in zip(df_weight['race'], df_weight['two_year_recid']):
        tmp_result.append(str(i) + str(j))
    df_weight['tmp'] = tmp_result

    df_weight.loc[df_weight['priors_count=missing'] == 1, 'is_missing'] = 1

    for i in df_weight['tmp'].unique():
        df_weight.loc[(df_weight['tmp'] == i) & (df_weight['is_missing'] == 0),
                      'weight'] = len(df_weight.loc[(df_weight['tmp'] == i),
                                                    :].index) / len(df_weight.loc[(df_weight['tmp'] == i) & (df_weight['is_missing'] == 0),
                                                                                  :].index)
        df_weight.loc[(df_weight['tmp'] == i) & (df_weight['is_missing'] == 1),
                      'weight'] = len(df_weight.loc[(df_weight['tmp'] == i) & (df_weight['is_missing'] == 0),
                                                    :].index) / len(df_weight.loc[(df_weight['tmp'] == i),
                                                                                  :].index)
    return np.array(df_weight['weight'])

def get_evaluation(dataset_orig_vt,y_pred,privileged_groups,unprivileged_groups,unpriv_val,priv_val,pos_label):
    print('Accuracy')
    print(accuracy_score(dataset_orig_vt.labels, y_pred))
    dataset_orig_vt_copy1 = dataset_orig_vt.copy()
    dataset_orig_vt_copy1.labels = y_pred

    metric_transf_train1 = BinaryLabelDatasetMetric(
        dataset_orig_vt_copy1,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups)
    print('p-rule')
    print(min(metric_transf_train1.disparate_impact(),
              1 / metric_transf_train1.disparate_impact()))
    print('FPR for unpriv group')
    orig_sens_att = dataset_orig_vt.protected_attributes.ravel()
    print(1 - TNR(dataset_orig_vt.labels.ravel()
                  [orig_sens_att == unpriv_val], y_pred[orig_sens_att == unpriv_val], pos_label))
    print("FNR for unpriv group")
    print(1 - TPR(dataset_orig_vt.labels.ravel()
                  [orig_sens_att == unpriv_val], y_pred[orig_sens_att == unpriv_val], pos_label))

    print('FPR for priv group')
    orig_sens_att = dataset_orig_vt.protected_attributes.ravel()
    print(1 - TNR(dataset_orig_vt.labels.ravel()
                  [orig_sens_att == priv_val], y_pred[orig_sens_att == priv_val], pos_label))
    print("FNR for priv group")
    print(1 - TPR(dataset_orig_vt.labels.ravel()
                  [orig_sens_att == priv_val], y_pred[orig_sens_att == priv_val], pos_label))


    

def get_distortion_compas_sel(vold, vnew):
    """Distortion function for the compas dataset. We set the distortion
    metric here. See section 4.3 in supplementary material of
    http://papers.nips.cc/paper/6988-optimized-pre-processing-for-discrimination-prevention
    for an example

    Note:
        Users can use this as templates to create other distortion functions.

    Args:
        vold (dict) : {attr:value} with old values
        vnew (dict) : dictionary of the form {attr:value} with new values

    Returns:
        d (value) : distortion value
    """
    # Distortion cost
    distort = {}
    distort['two_year_recid'] = pd.DataFrame(
        {'No recid.': [0., 2.],
         'Did recid.': [2., 0.]},
        index=['No recid.', 'Did recid.'])
    distort['age_cat'] = pd.DataFrame(
        {'Less than 25': [0., 1., 2.],
         '25 to 45': [1., 0., 1.],
         'Greater than 45': [2., 1., 0.]},
        index=['Less than 25', '25 to 45', 'Greater than 45'])
    distort['c_charge_degree'] = pd.DataFrame(
        {'M': [0., 2.],
         'F': [1., 0.]},
        index=['M', 'F'])
    distort['priors_count'] = pd.DataFrame(
        {'0': [0., 1., 2.],
         '1 to 3': [1., 0., 1.],
         'More than 3': [2., 1., 0.]},
        index=['0', '1 to 3', 'More than 3'])

    distort['score_text'] = pd.DataFrame(
        {'Low': [0., 2.],
         'MediumHigh': [2., 0.]},
        index=['Low', 'MediumHigh'])
    distort['sex'] = pd.DataFrame(
        {0.0: [0., 2.],
         1.0: [2., 0.]},
        index=[0.0, 1.0])
    distort['race'] = pd.DataFrame(
        {0.0: [0., 2.],
         1.0: [2., 0.]},
        index=[0.0, 1.0])

    total_cost = 0.0
    for k in vold:
        if k in vnew:
            total_cost += distort[k].loc[vnew[k], vold[k]]

    return total_cost


class CompasDataset_test(StandardDataset):
    def __init__(
            self,
            label_name='two_year_recid',
            favorable_classes=[0],
            protected_attribute_names=[
                'sex',
                'race'],
            privileged_classes=[
                ['Female'],
                ['Caucasian']],
            instance_weights_name=None,
            categorical_features=[
                'age_cat',
                'c_charge_degree',
                'c_charge_desc'],
            features_to_keep=[
                'sex',
                'age',
                'age_cat',
                'race',
                'juv_fel_count',
                'juv_misd_count',
                'juv_other_count',
                'priors_count',
                'c_charge_degree',
                'c_charge_desc',
                'two_year_recid',
                'length_of_stay'],
            features_to_drop=[],
            na_values=[],
            custom_preprocessing=default_preprocessing,
            metadata=default_mappings):
        np.random.seed(1)

        def quantizePrior1(x):
            if x <= 0:
                return 0
            elif 1 <= x <= 3:
                return 1
            else:
                return 2

        def quantizeLOS(x):
            if x <= 7:
                return 0
            if 8 < x <= 93:
                return 1
            else:
                return 2

        def group_race(x):
            if x == "Caucasian":
                return 1.0
            else:
                return 0.0

        filepath = 'data/compas/compas-test.csv'
        df = pd.read_csv(filepath, index_col='id', na_values=[])

        df['age_cat'] = df['age_cat'].replace('Greater than 45', 2)
        df['age_cat'] = df['age_cat'].replace('25 - 45', 1)
        df['age_cat'] = df['age_cat'].replace('Less than 25', 0)
        df['score_text'] = df['score_text'].replace('High', 1)
        df['score_text'] = df['score_text'].replace('Medium', 1)
        df['score_text'] = df['score_text'].replace('Low', 0)
        df['priors_count'] = df['priors_count'].apply(
            lambda x: quantizePrior1(x))
        df['length_of_stay'] = (pd.to_datetime(df['c_jail_out']) -
                                pd.to_datetime(df['c_jail_in'])).apply(
            lambda x: x.days)
        df['length_of_stay'] = df['length_of_stay'].apply(
            lambda x: quantizeLOS(x))
        df = df.loc[~df['race'].isin(
            ['Native American', 'Hispanic', 'Asian', 'Other']), :]
        df['c_charge_degree'] = df['c_charge_degree'].replace({'F': 0, 'M': 1})

        # _,df = train_test_split(df,test_size = 4000,random_state = 1)

        df['c_charge_degree'] = df['c_charge_degree'].replace({0: 'F', 1: 'M'})

        super(
            CompasDataset_test,
            self).__init__(
            df=df,
            label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop,
            na_values=na_values,
            custom_preprocessing=custom_preprocessing,
            metadata=metadata)

def load_preproc_data_compas_test(protected_attributes=None):
    def custom_preprocessing(df):
        df = df[['age',
                 'c_charge_degree',
                 'race',
                 'age_cat',
                 'score_text',
                 'sex',
                 'priors_count',
                 'days_b_screening_arrest',
                 'decile_score',
                 'is_recid',
                 'two_year_recid',
                 'length_of_stay']]

        # Indices of data samples to keep
        ix = df['days_b_screening_arrest'] <= 30
        ix = (df['days_b_screening_arrest'] >= -30) & ix
        ix = (df['is_recid'] != -1) & ix
        ix = (df['c_charge_degree'] != "O") & ix
        ix = (df['score_text'] != 'N/A') & ix
        df = df.loc[ix, :]

        # Restrict races to African-American and Caucasian
        dfcut = df.loc[~df['race'].isin(
            ['Native American', 'Hispanic', 'Asian', 'Other']), :]

        # Restrict the features to use
        dfcutQ = dfcut[['sex',
                        'race',
                        'age_cat',
                        'c_charge_degree',
                        'score_text',
                        'priors_count',
                        'is_recid',
                        'two_year_recid',
                        'length_of_stay']].copy()

        # Quantize priors count between 0, 1-3, and >3
        def quantizePrior(x):
            if x == 0:
                return '0'
            elif x == 1:
                return '1 to 3'
            elif x == 2:
                return 'More than 3'
            else:
                return 'missing'
        # Quantize length of stay

        def quantizeLOS(x):
            if x == 0:
                return '<week'
            if x == 1:
                return '<3months'
            else:
                return '>3 months'

        # Quantize length of stay
        def adjustAge(x):
            if x == 1:
                return '25 to 45'
            elif x == 2:
                return 'Greater than 45'
            elif x == 0:
                return 'Less than 25'
        # Quantize score_text to MediumHigh

        def quantizeScore(x):
            if x == 1:
                return 'MediumHigh'
            else:
                return 'Low'

        def group_race(x):
            if x == "Caucasian":
                return 1.0
            else:
                return 0.0

        dfcutQ['priors_count'] = dfcutQ['priors_count'].apply(
            lambda x: quantizePrior(x))
        dfcutQ['length_of_stay'] = dfcutQ['length_of_stay'].apply(
            lambda x: quantizeLOS(x))
        dfcutQ['score_text'] = dfcutQ['score_text'].apply(
            lambda x: quantizeScore(x))
        dfcutQ['age_cat'] = dfcutQ['age_cat'].apply(lambda x: adjustAge(x))
        # Recode sex and race
        dfcutQ['sex'] = dfcutQ['sex'].replace({'Female': 1.0, 'Male': 0.0})
        dfcutQ['race'] = dfcutQ['race'].apply(lambda x: group_race(x))

        features = ['two_year_recid', 'race',
                    'age_cat', 'priors_count', 'c_charge_degree', 'score_text']

        # Pass vallue to df
        df = dfcutQ[features]

        return df

    XD_features = [
        'age_cat',
        'c_charge_degree',
        'priors_count',
        'race',
        'score_text']
    D_features = [
        'race'] if protected_attributes is None else protected_attributes
    Y_features = ['two_year_recid']
    X_features = list(set(XD_features) - set(D_features))
    categorical_features = [
        'age_cat',
        'priors_count',
        'c_charge_degree',
        'score_text']

    # privileged classes
    all_privileged_classes = {"sex": [1.0],
                              "race": [1.0]}

    # protected attribute maps
    all_protected_attribute_maps = {
        "sex": {
            0.0: 'Male', 1.0: 'Female'}, "race": {
            1.0: 'Caucasian', 0.0: 'Not Caucasian'}}

    return CompasDataset_test(
        label_name=Y_features[0],
        favorable_classes=[0],
        protected_attribute_names=D_features,
        privileged_classes=[all_privileged_classes[x] for x in D_features],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features + Y_features + D_features,
        na_values=[],
        metadata={'label_maps': [{1.0: 'Did recid.', 0.0: 'No recid.'}],
                  'protected_attribute_maps': [all_protected_attribute_maps[x]
                                               for x in D_features]},
        custom_preprocessing=custom_preprocessing)

class CompasDataset_train(StandardDataset):
    def __init__(
            self,
            label_name='two_year_recid',
            favorable_classes=[0],
            protected_attribute_names=[
                'sex',
                'race'],
            privileged_classes=[
                ['Female'],
                ['Caucasian']],
            instance_weights_name=None,
            categorical_features=[
                'age_cat',
                'c_charge_degree',
                'c_charge_desc'],
            features_to_keep=[
                'sex',
                'age',
                'age_cat',
                'race',
                'juv_fel_count',
                'juv_misd_count',
                'juv_other_count',
                'priors_count',
                'c_charge_degree',
                'c_charge_desc',
                'two_year_recid',
                'length_of_stay'],
            features_to_drop=[],
            na_values=[],
            custom_preprocessing=default_preprocessing,
            metadata=default_mappings):
        np.random.seed(1)

        def quantizePrior1(x):
            if x <= 0:
                return 0
            elif 1 <= x <= 3:
                return 1
            else:
                return 2

        def quantizeLOS(x):
            if x <= 7:
                return 0
            if 8 < x <= 93:
                return 1
            else:
                return 2

        def group_race(x):
            if x == "Caucasian":
                return 1.0
            else:
                return 0.0

        filepath = 'data/compas/compas-train.csv'
        df = pd.read_csv(filepath, index_col='id', na_values=[])

        df['age_cat'] = df['age_cat'].replace('Greater than 45', 2)
        df['age_cat'] = df['age_cat'].replace('25 - 45', 1)
        df['age_cat'] = df['age_cat'].replace('Less than 25', 0)
        df['score_text'] = df['score_text'].replace('High', 1)
        df['score_text'] = df['score_text'].replace('Medium', 1)
        df['score_text'] = df['score_text'].replace('Low', 0)
        df['priors_count'] = df['priors_count'].apply(
            lambda x: quantizePrior1(x))
        df['length_of_stay'] = (pd.to_datetime(df['c_jail_out']) -
                                pd.to_datetime(df['c_jail_in'])).apply(
            lambda x: x.days)
        df['length_of_stay'] = df['length_of_stay'].apply(
            lambda x: quantizeLOS(x))
        df = df.loc[~df['race'].isin(
            ['Native American', 'Hispanic', 'Asian', 'Other']), :]
        df['c_charge_degree'] = df['c_charge_degree'].replace({'F': 0, 'M': 1})

        ix = df['days_b_screening_arrest'] <= 30
        ix = (df['days_b_screening_arrest'] >= -30) & ix
        ix = (df['is_recid'] != -1) & ix
        ix = (df['c_charge_degree'] != "O") & ix
        ix = (df['score_text'] != 'N/A') & ix
        df = df.loc[ix, :]
        df['c_charge_degree'] = df['c_charge_degree'].replace({0: 'F', 1: 'M'})

        super(
            CompasDataset_train,
            self).__init__(
            df=df,
            label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop,
            na_values=na_values,
            custom_preprocessing=custom_preprocessing,
            metadata=metadata)
def load_preproc_data_compas_test_comb(protected_attributes=None):
    def custom_preprocessing(df):
        """The custom pre-processing function is adapted from
            https://github.com/fair-preprocessing/nips2017/blob/master/compas/code/Generate_Compas_Data.ipynb
        """

        df = df[['age',
                 'c_charge_degree',
                 'race',
                 'age_cat',
                 'score_text',
                 'sex',
                 'priors_count',
                 'days_b_screening_arrest',
                 'decile_score',
                 'is_recid',
                 'two_year_recid',
                 'length_of_stay']]

        # Indices of data samples to keep
        ix = df['days_b_screening_arrest'] <= 30
        ix = (df['days_b_screening_arrest'] >= -30) & ix
        ix = (df['is_recid'] != -1) & ix
        ix = (df['c_charge_degree'] != "O") & ix
        ix = (df['score_text'] != 'N/A') & ix
        df = df.loc[ix, :]

        # Restrict races to African-American and Caucasian
        dfcut = df.loc[~df['race'].isin(
            ['Native American', 'Hispanic', 'Asian', 'Other']), :]

        # Restrict the features to use
        dfcutQ = dfcut[['sex',
                        'race',
                        'age_cat',
                        'c_charge_degree',
                        'score_text',
                        'priors_count',
                        'is_recid',
                        'two_year_recid',
                        'length_of_stay']].copy()

        # Quantize priors count between 0, 1-3, and >3
        def quantizePrior(x):
            if x == 0:
                return '0'
            elif x == 1:
                return '1 to 3'
            elif x == 2:
                return 'More than 3'
            else:
                return 'missing'
        # Quantize length of stay

        def quantizeLOS(x):
            if x == 0:
                return '<week'
            if x == 1:
                return '<3months'
            else:
                return '>3 months'

        # Quantize length of stay
        def adjustAge(x):
            if x == 1:
                return '25 to 45'
            elif x == 2:
                return 'Greater than 45'
            elif x == 0:
                return 'Less than 25'
        # Quantize score_text to MediumHigh

        def quantizeScore(x):
            if x == 1:
                return 'MediumHigh'
            else:
                return 'Low'

        def group_race(x):
            if x == "Caucasian":
                return 1.0
            else:
                return 0.0

        dfcutQ['priors_count'] = dfcutQ['priors_count'].apply(
            lambda x: quantizePrior(x))
        dfcutQ['length_of_stay'] = dfcutQ['length_of_stay'].apply(
            lambda x: quantizeLOS(x))
        dfcutQ['score_text'] = dfcutQ['score_text'].apply(
            lambda x: quantizeScore(x))
        dfcutQ['age_cat'] = dfcutQ['age_cat'].apply(lambda x: adjustAge(x))
        # Recode sex and race
        dfcutQ['sex'] = dfcutQ['sex'].replace({'Female': 1.0, 'Male': 0.0})
        dfcutQ['race'] = dfcutQ['race'].apply(lambda x: group_race(x))

        features = ['two_year_recid', 'race',
                    'age_cat', 'priors_count', 'c_charge_degree', 'score_text']

        # Pass vallue to df
        df = dfcutQ[features]
        df['mis_prob'] = 0
        for index, row in df.iterrows():
            if row['race'] != 'African-American' and row['two_year_recid']==0:
                df.loc[index, 'mis_prob'] = 0.3
            elif row['race'] != 'African-American':
                df.loc[index, 'mis_prob'] = 0.1
            else:
                df.loc[index, 'mis_prob'] = 0.05
        new_label = []
        for index, row in df.iterrows():
            if np.random.binomial(1, float(row['mis_prob']), 1)[0] == 1:
                new_label.append('missing')
            else:
                new_label.append(row['priors_count'])
        df['priors_count'] = new_label

        return df

    XD_features = [
        'age_cat',
        'c_charge_degree',
        'priors_count',
        'race',
        'score_text']
    D_features = [
        'race'] if protected_attributes is None else protected_attributes
    Y_features = ['two_year_recid']
    X_features = list(set(XD_features) - set(D_features))
    categorical_features = [
        'age_cat',
        'priors_count',
        'c_charge_degree',
        'score_text']

    # privileged classes
    all_privileged_classes = {"sex": [1.0],
                              "race": [1.0]}

    # protected attribute maps
    all_protected_attribute_maps = {
        "sex": {
            0.0: 'Male', 1.0: 'Female'}, "race": {
            1.0: 'Caucasian', 0.0: 'Not Caucasian'}}

    return CompasDataset_test(
        label_name=Y_features[0],
        favorable_classes=[0],
        protected_attribute_names=D_features,
        privileged_classes=[all_privileged_classes[x] for x in D_features],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features + Y_features + D_features,
        na_values=[],
        metadata={'label_maps': [{1.0: 'Did recid.', 0.0: 'No recid.'}],
                  'protected_attribute_maps': [all_protected_attribute_maps[x]
                                               for x in D_features]},
        custom_preprocessing=custom_preprocessing)