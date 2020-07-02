import pandas as pd
import numpy as np
from functools import reduce

from navara.utils import log_step


def read_data(data_path):
    """
    Get data by specifying a datapath where the data is stored.
    The different dataframes will be read first and subsequently, the duplicates will be dropped.
    :param data_path: data path of the CSV file
    :return: dataframe
    """

    df_1 = pd.read_csv(f'{data_path}/data_1.csv').drop_duplicates(subset=['codering'])
    df_2 = pd.read_csv(f'{data_path}/data_2.csv').drop_duplicates(subset=['codering'])
    df_3 = pd.read_csv(f'{data_path}/data_3.csv').drop_duplicates(subset=['codering'])

    dfs = [df_1, df_2, df_3]

    return reduce(lambda left, right: pd.merge(left, right, on=['codering', 'Unnamed: 0', 'gemeentenaam']), dfs)


def drop_irrelevant_features(df):
    """
    This function drops irrevalant features that are not required in the model.
    :param df: dataframe
    :return: dataframe
    """
    cols = ['Unnamed: 0',
            'id',
            'codering',
            'woningkenmerken',
            'id_old']

    return df.drop(columns=list(cols))


def drop_rows(df):
    """
    This function drops the entire rows where the amount of inhabitants is less than 10.
    :param df: dataframe
    :return: dataframe
    """
    return df.drop(df[abs(df['aantal_inwoners'])<10].index)


def drop_missing_values_columns(df):
    """
    This function drops features that contain too many missing values.
    :param df: dataframe
    :return: dataframe
    """
    cols = ['elect_appartement',
            'elect_tussenwoning',
            'elect_hoekwoning',
            'elect_twee_onder_een_kap_woning',
            'aard_appartement',
            'aard_tussenwoning',
            'aard_hoekwoning',
            'aard_twee_onder_een_kap_woning',
            'percentage_woningen_met_stadsverwarming',
            'gemiddeld_inkomen_per_inkomensontvanger',
            'gemiddeld_inkomen_per_inwoner']

    return df.drop(columns=list(cols))


def make_numeric_features_absolute(df):
    """
    This function makes all numeric features absolute.
    :param df: dataframe
    :return: dataframe
    """
    return df.apply(lambda d: d.abs() if np.issubdtype(d.dtype, np.number) else d)


def transform_skewed_data(df):
    """
    This function transforms certain columns to create a more normal or symmetric distribution.
    :param df: dataframe
    :return: dataframe
    """
    return df.assign(
        aantal_inwoners = lambda d: np.log1p(d['aantal_inwoners']),
        stadsverwarming=lambda d: np.log1p(d['stadsverwarming']),
        mannen=lambda d: np.log1p(d['mannen']),
        vrouwen=lambda d: np.log1p(d['vrouwen']),
        k_0_tot_15_jaar=lambda d: np.log1p(d['k_0_tot_15_jaar']),
        k_15_tot_25_jaar=lambda d: np.log1p(d['k_15_tot_25_jaar']),
        k_25_tot_45_jaar=lambda d: np.log1p(d['k_25_tot_45_jaar']),
        k_45_tot_65_jaar=lambda d: np.log1p(d['k_45_tot_65_jaar']),
        k_65_jaar_of_ouder=lambda d: np.log1p(d['k_65_jaar_of_ouder']),
        huishoudens_totaal=lambda d: np.log1p(d['huishoudens_totaal']),
        eenpersoonshuishoudens=lambda d: np.log1p(d['eenpersoonshuishoudens']),
        huishoudens_zonder_kinderen=lambda d: np.log1p(d['huishoudens_zonder_kinderen']),
        huishoudens_met_kinderen=lambda d: np.log1p(d['huishoudens_met_kinderen']),
        gemiddelde_huishoudensgrootte=lambda d: np.log(d['gemiddelde_huishoudensgrootte']),
        bevolkingsdichtheid=lambda d: np.log1p(d['bevolkingsdichtheid']),
        woningvoorraad=lambda d: np.log1p(d['woningvoorraad']),
        gemiddelde_woningwaarde=lambda d: np.log(d['gemiddelde_woningwaarde']),
        in_bezit_woningcorporatie=lambda d: np.log1p(d['in_bezit_woningcorporatie']),
        in_bezit_overige_verhuurders=lambda d: np.log1p(d['in_bezit_overige_verhuurders']),
        eigendom_onbekend=lambda d: np.log1p(d['eigendom_onbekend']),
        bouwjaar_voor_2000=lambda d: np.log((d['bouwjaar_voor_2000'].max()+1) - d['bouwjaar_voor_2000']),
        bouwjaar_vanaf_2000=lambda d: np.log1p(d['bouwjaar_vanaf_2000']),
        gemiddeld_elektriciteitsverbruik_totaal=lambda d: np.sqrt(d['gemiddeld_elektriciteitsverbruik_totaal']),
        elect_huurwoning=lambda d: np.log(d['elect_huurwoning']),
        gemiddeld_aardgasverbruik_totaal=lambda d: np.sqrt(d['gemiddeld_aardgasverbruik_totaal']),
        aantal_inkomensontvangers=lambda d: np.log1p(d['aantal_inkomensontvangers']),
        k_40_huishoudens_met_laagste_inkomen=lambda d: np.sqrt(d['k_40_huishoudens_met_laagste_inkomen']),
        k_20_huishoudens_met_hoogste_inkomen=lambda d: np.sqrt(d['k_20_huishoudens_met_hoogste_inkomen']),
        personen_per_soort_uitkering_bijstand=lambda d: np.log1p(d['personen_per_soort_uitkering_bijstand']),
        personenautos_brandstof_benzine=lambda d: np.log1p(d['personenautos_brandstof_benzine']),
        personenautos_overige_brandstof=lambda d: np.log1p(d['personenautos_overige_brandstof']),
        oppervlakte_land=lambda d: np.log1p(d['oppervlakte_land']),
        omgevingsadressendichtheid=lambda d: np.log1p(d['omgevingsadressendichtheid']),
        bedrijfsvestigingen_totaal=lambda d: np.log1p(d['bedrijfsvestigingen_totaal']),
        type_a_landbouw_bosbouw_visserij=lambda d: np.log1p(d['type_a_landbouw_bosbouw_visserij']),
        type_bf_nijverheid_energie=lambda d: np.log1p(d['type_bf_nijverheid_energie']),
        type_gi_handel_horeca=lambda d: np.log1p(d['type_gi_handel_horeca']),
        type_hj_vervoer_informatie_communicatie=lambda d: np.log1p(d['type_hj_vervoer_informatie_communicatie']),
        type_kl_financiele_diensten_onroerendgoed=lambda d: np.log1p(d['type_kl_financiele_diensten_onroerendgoed']),
        type_mn_zakelijke_dienstverlening=lambda d: np.log1p(d['type_mn_zakelijke_dienstverlening']),
        type_ru_cultuur_recreatie_overige_diensten=lambda d: np.log1p(d['type_ru_cultuur_recreatie_overige_diensten']),
        aantal_installaties_bij_woningen=lambda d: np.log1p(d['aantal_installaties_bij_woningen']),
        aantal_zonnepanelen_per_installatie=lambda d: np.log1p(d['aantal_zonnepanelen_per_installatie']),
        opgesteld_vermogen_van_zonnepanelen=lambda d: np.log1p(d['opgesteld_vermogen_van_zonnepanelen']),
        totaal_aantal_laadpalen=lambda d: np.log1p(d['totaal_aantal_laadpalen']),
        werkloosheidsuitkering_relatief=lambda d: np.log1p(d['werkloosheidsuitkering_relatief']),
        bijstandsuitkering_relatief=lambda d: np.log1p(d['bijstandsuitkering_relatief']),
        arbeidsongeschiktheidsuitkering_relatief=lambda d: np.log1p(d['arbeidsongeschiktheidsuitkering_relatief']),
        inwoners_vanaf_15_jaar=lambda d: np.log1p(d['inwoners_vanaf_15_jaar']),
        inwoners_vanaf_15_jr_tot_aow_leeftijd=lambda d: np.log1p(d['inwoners_vanaf_15_jr_tot_aow_leeftijd']),
        inwoners_vanaf_de_aow_leeftijd=lambda d: np.log1p(d['inwoners_vanaf_de_aow_leeftijd'])
    )


def create_categorical_combinations(df):
    """
    This function creates combinations of two categorical features, as input for feature hashing in the machine learning
    pipeline.
    :param df: dataframe
    :return: dataframe
    """
    return df.assign(
        gemeentenaam_regio=lambda d: d['gemeentenaam']+d['soort_regio']
    )


@log_step
def get_df(data_path):
    return (read_data(data_path)
            .pipe(drop_irrelevant_features)
            .pipe(drop_rows)
            .pipe(drop_missing_values_columns)
            .pipe(make_numeric_features_absolute)
            .pipe(transform_skewed_data)
            .pipe(create_categorical_combinations)
            ).drop(columns=['gemeentenaam',
                            'soort_regio'])


@log_step
def get_original_df(data_path):
    return (read_data(data_path)
            .pipe(drop_irrelevant_features)
            .pipe(drop_rows)
            .pipe(drop_missing_values_columns)
            .pipe(make_numeric_features_absolute)
            )