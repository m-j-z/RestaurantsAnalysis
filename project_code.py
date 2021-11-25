import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


# =============================================================================
# 'chain' is a list of x number of arrays for number of chain restaurants
# 'independent' is a list of x number of arrays for number of independent restaurants
# 'coordinates' is a list of x number of arrays for number of coordinates
# Plot the density of the number of restaurants at the coordinates
# =============================================================================
def chain_independent_plot(chain, independent, coordinates):
    num_plots = len(chain)
    
    for x in range(num_plots):
        chain_y = chain[x].tolist()
        independent_y = independent[x].tolist()
        plt.figure()
        y = range(len(chain_y))
        plt.xticks(y, coordinates[x], rotation='vertical')
        plt.xlabel('Coordinates (latitude, longitude)')
        plt.ylabel('Density (restaurants/km)')
        plt.title('Number of Restaurants Per km at the Given Coordinate')
        plt.plot(y, chain_y, 'o')
        plt.plot(y, independent_y, 'o')
        plt.legend(['Chain Restaurants', 'Independent Restaurants'])
        plt.savefig('chain-independent_' + str(x) + '.png', bbox_inches='tight')
        # plt.show()


# ============================================================================
# 'df' a dataframe with data of all places in greater vancouver
# Extract the density of chain and independent restaurants at a given coordinate
# =============================================================================
def chain_independent_density(df):
    # tags to define restaurants
    tags = ['cafe', 'fast_food', 'pub', 'restaurant']
    pattern = '|'.join(tags)
    # create new dataframe with necessary data
    df_density = df[['lat', 'lon', 'amenity', 'name']].copy()
    df_density = df_density[df_density['amenity'].str.fullmatch(pattern)]
    
    # chain restaurants have at least 10 occurances in the data
    # non-chain restaurants have at most 9 occurances in the data
    chain = df_density.groupby(['name']).filter(lambda x: len(x) >= 10).reset_index()
    independent = df_density.groupby(['name']).filter(lambda x: len(x) < 10).reset_index()
    chain = chain.drop('index', axis = 1)
    independent = independent.drop('index', axis = 1)
    
    # round to nearest 0.0x
    chain['lat'] = chain['lat'] - 0.004
    chain['lon'] = chain['lon'] - 0.004
    chain['lat'] = chain['lat'].round(2)
    chain['lon'] = chain['lon'].round(2)
    independent['lat'] = independent['lat'] - 0.004
    independent['lon'] = independent['lon'] - 0.004
    independent['lat'] = independent['lat'].round(2)
    independent['lon'] = independent['lon'].round(2)
    
    # group chain and independent restaurants by coordinates
    chain = chain.groupby(['lat', 'lon'])
    independent = independent.groupby(['lat', 'lon'])
    chain_dfs = [chain.get_group(x) for x in chain.groups]
    independent_dfs = [independent.get_group(x) for x in independent.groups]
    
    # find and append the number of restaurants for each matching coordinate
    chain_density = []
    independent_density = []
    coordinates = []
    for chain_df in chain_dfs:
        for independent_df in independent_dfs:
            if chain_df['lat'].iloc[0] == independent_df['lat'].iloc[0] and chain_df['lon'].iloc[0] == independent_df['lon'].iloc[0]:
                chain_density.append(chain_df.shape[0])
                independent_density.append(independent_df.shape[0])
                coordinates.append([chain_df['lat'].iloc[0], chain_df['lon'].iloc[0]])
            
    # restaurants / km
    chain_density = [x / 1.11 for x in chain_density]
    independent_density = [x / 1.11 for x in independent_density]
    
    _, p = mannwhitneyu(chain_density, independent_density)
    print('The p-value is:', p)
    
    # split lists
    max_pts = 20
    num_splits = int(round((len(chain_density) / max_pts), 1))
    
    np_chain_density = np.array_split(np.array(chain_density), num_splits)
    np_independent_density = np.array_split(np.array(independent_density), num_splits)
    np_coordinates = np.array_split(np.array(coordinates), num_splits)
    
    chain_independent_plot(np_chain_density, np_independent_density, np_coordinates)


# =============================================================================
# 'dfs' is list of dataframe
# 'name' is name of restaurant to compare
# sees if there are more restaurants in certain areas
# =============================================================================
def more_restaurants(dfs, name):
    
    contains_name = []
    for df in dfs:
        tmp = df[df['name'].str.fullmatch(name)].reset_index()
        tmp = tmp.drop('index', axis = 1)
        if tmp is not None and not tmp.empty:
            contains_name.append(tmp)
        
    # remove empty dataframes
    contains_name = list(filter(lambda df: not df.empty, contains_name))
    
    # concat all dataframes
    if not contains_name:
        return
    contains_name = pd.concat(contains_name)
    
    # calculate std_dev and mean
    std_dev = contains_name['count'].std()
    mean = contains_name['count'].mean()
    
    # get outliers, those that are > 3 std dev from mean
    outlier = mean + 3 * std_dev
    outliers = contains_name[contains_name['count'] > outlier]
    
    # concat outliers
    outliers = outliers.reset_index()
    outliers = outliers.drop('index', axis = 1)
    
    return outliers


# =============================================================================
# df is dataframe
# number of restaurants to nearest 0.01 of a degree or 1.11km (area)
# =============================================================================
def chain_restaurant_difference(df):
    
    # tags to define restaurants
    tags = ['cafe', 'fast_food', 'pub', 'restaurant']
    pattern = '|'.join(tags)
    # create new dataframe with necessary data
    df_diff = df[['lat', 'lon', 'amenity', 'name']].copy()
    df_diff = df_diff[df_diff['amenity'].str.fullmatch(pattern)]
    
    # chain restaurants have at least 10 occurances in the data
    chain_diff = df_diff.groupby(['name']).filter(lambda x: len(x) >= 10).reset_index()
    
    # round and cull NaN
    chain_diff['lat'] = chain_diff['lat'] - 0.004
    chain_diff['lon'] = chain_diff['lon'] - 0.004
    chain_diff['lat'] = chain_diff['lat'].round(2)
    chain_diff['lon'] = chain_diff['lon'].round(2)
    
    # group by latitude and longitude
    chain_diff = chain_diff.dropna()
    chain_diff = chain_diff.groupby(['lat', 'lon'])
    
    # get group for each latitude
    chain_dfs = [chain_diff.get_group(x) for x in chain_diff.groups]
    
    # fix up data, returns as list of df
    comparison_data = []
    for chain_df in chain_dfs:
        if chain_df.empty:
            continue
        chain_tmp = chain_df.groupby(['lat', 'lon', 'name']).size().reset_index(name='count')
        tmp = []
        for row in chain_tmp.itertuples():
            tmp.append([row.lat, row.lon, row.name, row.count])
        comparison_data.append(tmp.copy())
    
    list_df = []
    for x in comparison_data:
        tmp = pd.DataFrame(x, columns=['lat', 'lon', 'name', 'count'])
        list_df.append(tmp)
    
    return list_df


def get_all_restaurants(df):
    
    # tags to define restaurants
    tags = ['cafe', 'fast_food', 'pub', 'restaurant']
    pattern = '|'.join(tags)
    # create new dataframe with necessary data
    df_names = df[['lat', 'lon', 'amenity', 'name']].copy()
    df_names = df_names[df_names['amenity'].str.fullmatch(pattern)]
    
    chain_names = df_names.groupby(['name']).filter(lambda x: len(x) >= 10).reset_index()
    
    restaurants = chain_names['name'].unique()
    return restaurants
    


# =============================================================================
# Accepts 2 arguments, first is file name and second is which restaurant to look at
# =============================================================================
def main():
    
    file_name = 'amenities-vancouver.json'
    look_at = None
    
    if len(sys.argv) == 2:
        arg1 = sys.argv[1]
        if '.json' in arg1:
            file_name = arg1
        elif '.csv' in arg1:
            look_at = pd.read_csv(arg1)
    elif len(sys.argv) == 3:
        file_name = sys.argv[1]
        look_at = sys.argv[2]
    elif len(sys.argv) > 3:
        print('Accepts only 2 arguments.')
        return
    
    data = pd.read_json(file_name, lines = True)
    data = data.drop('timestamp', axis = 1)
    chain_diff_dfs = chain_restaurant_difference(data)
    if look_at is None:
        look_at = get_all_restaurants(data)
        
    violating_chains = []
    count = 0
    for restraunt in look_at:
        ret_restraunt = more_restaurants(chain_diff_dfs, restraunt)
        if ret_restraunt is not None and not ret_restraunt.empty:
            count += 1
            violating_chains.append(ret_restraunt)
    
    print('From the analysis, there is', count, 'chain restaurants that have more stores in certain places than others.')
    finished_data = pd.concat(violating_chains, ignore_index = True)
    finished_data.to_csv('analysis_result.csv', index=False)
    
    chain_independent_density(data)
    
    
if __name__ == '__main__':
    main()