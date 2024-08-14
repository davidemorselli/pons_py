import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import itertools
from sklearn.metrics import silhouette_samples
from KDEpy import FFTKDE
from .visualisation import Visualisation
from gensim.models import Word2Vec, KeyedVectors
from random import seed, shuffle
from gensim.models.callbacks import CallbackAny2Vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from tqdm import tqdm
tqdm.pandas()
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
from scipy.stats import pearsonr


class Calculation:
    def __init__(self, data):
        self.data = data 

    def silhouette(self, coord_columns, group_column, subset=None, plot=False, cmap=None, plot_kwargs=None):
        """
        Calculate the silhouette scores for each group in the Speaker Landscape. This can be done for 1d or 2d data. If a subset is provided, then it will be their silhouette scores (not the entire dataset). You can choose which groups using the subset (list) parameter. You can also plot them using the plot parameter. Missing values will be removed from the calculation.

        Note: Due to the behaviour of Matplotlib, if you are running this from a Python script (rather than a Jupyter Notebook), the function will not return the results UNTIL you close the plot window. This is a limitation of Matplotlib.

        Parameters:
        coord_columns (str or list): The column name(s) of the 1d or 2d data.
        group_column (str): The column name of the group column.
        subset (list): A list of groups to calculate the silhouette scores for. If None, all groups will be used.
        plot (bool): Whether to plot the silhouette scores. Default is False.
        cmap (dict or str): A dictionary of colours for each group in the form of {group: color}, or a string with the name of a matplotlib Colormap. If None, the 'tab10' colormap will be used.
        plot_kwargs (dict): A dictionary of matplotlib arguments to pass to the plot

        Returns:
        dict: A dictionary of silhouette scores for each group in the form of {group: silhouette_score}. It also includes the proportion of silhouettes over 0 in the dataset as 'Sil_over_0'.

        """

        data = self.data
        if subset is None:
            subset = data
        else:
            subset = data[data[group_column].isin(subset)]
        
        if isinstance(coord_columns, str):
            data_columns = [coord_columns]
        else:
            data_columns = coord_columns

        if subset[data_columns].isna().sum().sum() > 0 or subset[group_column].isna().sum() > 0:
            print('WARNING: There are missing values in the data. They will removed to allow the calculation to proceed (this will not affect the original dataset).')
            cols = data_columns + [group_column]
            subset = subset.dropna(subset=cols)
        
        if subset[data_columns].shape[1] > 2 or subset[data_columns].shape[1] == 0:
            raise ValueError('The number of columns in data_columns must be 2 or 1.')
        elif subset[data_columns].shape[1] == 1:
            print('Only one column indicated - Calcualting 1d silhouette scores...')
            subset['silhouette'] = silhouette_samples(subset[data_columns].values.reshape(-1,1), subset[group_column])
            subset['norm_sil'] = subset.silhouette.apply(lambda x: (x+1)/2)
        else:
            print('Calculating 2d silhouette scores...')
            subset['silhouette'] = silhouette_samples(subset[data_columns], subset[group_column])
            subset['norm_sil'] = subset.silhouette.apply(lambda x: (x+1)/2)


        sil = {i: subset[subset[group_column] == i]['silhouette'].mean() for i in subset[group_column].unique()}
        norm_sil = {i: subset[subset[group_column] == i]['norm_sil'].mean() for i in subset[group_column].unique()}
        final_sil = {i: {'silhouette': sil[i], 'norm_silhouette': norm_sil[i]} for i in sil}
        final_sil['Sil_over_0'] = len(subset[subset['silhouette'] > 0])/len(subset)

        if plot:
            Visualisation(subset).silhouette(group_column=group_column, silhouette_column='silhouette', subset=subset[group_column].unique(), cmap=cmap, plot_kwargs=plot_kwargs)

        print('Done!')
        return final_sil
        # takes 1d or 2d data and group column, returning a dictionary of silhouette scores for each group.

    def overlap(self, col_1d, group_column, subset=None, plot=False, cmap=None, plot_kwargs=None):
        """
        Calculate the overlap of groups in a 1d Speaker Landscape. This is normalised by the proportion of the data they occupy. If a subset is provided, then it will be their proportion of the subset (not the entire dataset). You can choose which groups using the subset (list) parameter. You can also plot them using the plot parameter. Missing values will be removed from the calculation. 
        Note: Due to the behaviour of Matplotlib, if you are running this from a Python script (rather than a Jupyter Notebook), the function will not return the results UNTIL you close the plot window. This is a limitation of Matplotlib.
        
        Parameters:
        col_1d (str): The column name of the 1d data.
        group_column (str): The column name of the group column.
        subset (list): A list of groups to calculate the overlap for. If None, all groups will be used.
        plot (bool): Whether to plot the overlap. Default is False.
        cmap (dict or str): A dictionary of colours for each group in the form of {group: color}, or a string with the name of a matplotlib Colormap. If None, the 'tab10' colormap will be used.
        plot_kwargs (dict): A dictionary of matplotlib arguments to pass to the plot function.

        Returns:
        float: The overlap of groups. This goes from 0 (no overlap between groups) to 1 (complete overlap between groups).

        """


        # Define a function for KDE with Sheather-Jones bandwidth using KDEpy
        data = self.data
        def kde(data, x_grid, bw='ISJ'):
            kde = FFTKDE(bw=bw).fit(data.values).evaluate(x_grid)
            return kde

        def calculate_weighted_overlap(density_functions, x):
            n = len(density_functions)
            overlap_areas = {}
            
            # Compute the overlap for each combination
            for r in range(2, n+1):  # Start from 2
                for combo in itertools.combinations(density_functions, r):
                    min_overlap = np.minimum.reduce([densities[color] for color in combo])
                    overlap_areas[combo] = min_overlap
            total_overlap = np.zeros(len(x))
            covered_areas = np.zeros(len(x))

            # Add the highest order overlaps first
            for combo in itertools.combinations(density_functions, n):
                min_overlap = overlap_areas[combo]
                total_overlap += min_overlap * n
                covered_areas = np.maximum(covered_areas, min_overlap)

            # Add lower order overlaps, subtracting areas already covered
            for r in range(n-1, 1, -1):  # From n-1 down to 2
                for combo in itertools.combinations(density_functions, r):
                    min_overlap = overlap_areas[combo]
                    # Subtract the overlap that is already covered
                    not_covered_overlap = np.maximum(min_overlap - covered_areas, 0)
                    total_overlap += not_covered_overlap * r
                    covered_areas = np.maximum(covered_areas, min_overlap)
    
            return np.trapz(total_overlap, x)


        if subset is None:
            subset = data[[col_1d, group_column]]
        else:
            subset = data[data[group_column].isin(subset)]
            subset = subset[[col_1d, group_column]]
        
        if subset[col_1d].isna().sum() > 0 or subset[group_column].isna().sum() > 0:
            print('WARNING: There are missing values in the data. They will removed to allow the calculation to proceed (this will not affect the original dataset).')
            subset = subset.dropna(subset=[col_1d, group_column])
        # print(locals())
        # Determine the number of groups
        n_groups = len(subset[group_column].unique())
        print(f'Number of groups: {n_groups}')

        # Define the evaluation grid
        max_val = subset[col_1d].max()
        min_val = subset[col_1d].min() 
        x_grid = np.linspace(min_val, max_val, 100000)


        # Define a small increment for expanding the range
        increment = np.abs(subset[col_1d].max() - subset[col_1d].min()) * 0.01  # 1% of the initial range

        # Maximum number of allowed ValueErrors before stopping
        max_attempts = 10
        error_count = 0

        densities = {}
        while True:
            try:
                for cluster in subset[group_column].unique():
                    cluster_data = subset[subset[group_column] == cluster][col_1d]
                    kde_data = kde(cluster_data, x_grid, bw='ISJ')
                    densities[cluster] = kde_data
                break
                
            except ValueError as e:
                error_count += 1
                
                # Check if the maximum number of attempts has been reached
                if error_count >= max_attempts:
                    print(f"Stopping after {max_attempts} failed attempts to calculate KDE. This is probably due to missing values, but read the error below: \n")
                    print(e)
                    break  # Exit the loop
                
                # Expand the range and create a new x_grid
                max_val += increment
                min_val -= increment
                x_grid = np.linspace(min_val, max_val, 100000)


        total_density = sum(densities.values())
        for cluster in densities:
            densities[cluster] /= np.trapz(total_density, x_grid)

        density_functions = [color for color in densities.keys()]

        overlap = calculate_weighted_overlap(density_functions, x_grid)

        if plot:
            Visualisation().overlap(densities=densities, x_grid=x_grid, cmap=cmap, plot_kwargs=plot_kwargs)
        
        return overlap

    def sl_measures(self, col_1d, group_column, subset=None, plot=False, cmap=None, plot_kwargs=None):
        """
        Calculates polarisation measures in the Speaker Landscape. Only accepts 1d landscapes (even for silhouettes, if you want 2d silhouettes, call the silhouette method directly). Plotting of the silhouette and overlap is possible here, however it only works in Jupyter Notebooks (.ipynb files or Google Colab), in a standard Python script, it will return the measures only after you close the plot windows (this is due to matplotlib limitations).
        
        Parameters:
        col_1d (str): The column name of the 1d data.
        group_column (str): The column name of the group column.
        subset (list): A list of groups to calculate the measures for. If None, all groups will be used.
        plot (bool): Whether to plot the silhouette and overlap. Default is False. Only works on Jupyter notebooks.
        cmap (dict or str): A dictionary of colours for each group in the form of {group: color}, or a string with the name of a matplotlib Colormap. If None, the 'tab10' colormap will be used.
        plot_kwargs (dict): A dictionary of matplotlib arguments to pass to the plot function.
        
        Returns:
        dict: A dictionary of the following measures: 'groups' (list of groups), 'overlap', '1-Overlap', 'avg_silhouette', 'Sil_over_0', 'C1', 'C2'.


        Note: 
        C1 = avg_silhouette * (1 - overlap)
        C2 = 2 * (1 - (avg_silhouette * overlap) - 0.5)
        """

        if isinstance(subset, str):
            subset = [subset]


        if subset is None:
            data = self.data
        elif len(subset) == 2:
            print(f'Subset has 2 groups: {subset}. Calculating measures pairwise...')
            data = self.data[self.data[group_column].isin(subset)]
        elif len(subset) == 1:
            # print(subset)
            raise ValueError('Subset must have at least 2 groups.')
        else:
            print(f'Subset has {len(subset)} groups: {subset}. Calculating collective measures...')
            data = self.data[self.data[group_column].isin(subset)]


        obj = Calculation(data)
        overlap = obj.overlap(col_1d, group_column, subset, plot, cmap, plot_kwargs)
        sil = obj.silhouette(col_1d, group_column, subset, plot, cmap, plot_kwargs)

        
        one_minus_overlap = 1 - overlap
        norm_sils = [sil[i]['norm_silhouette'] for i in sil if i != 'Sil_over_0']
        sil_avg = np.mean(norm_sils)
        sil_over_0 = sil['Sil_over_0']

        c1 = sil_avg * one_minus_overlap
        c2 = 2 * (1 - (sil_avg * overlap) - 0.5)

        measures = {'groups': data[group_column].unique().tolist(), 'overlap': overlap, '1-Overlap': one_minus_overlap, 'avg_silhouette': sil_avg, 'Sil_over_0': sil_over_0, 'C1': c1, 'C2': c2}

        return measures

    # def sn_measures(self):
    #     pass
    #     # takes graph object, returning a dictionary of all polarisation measures of the SN.

    def term_correlations(self, embedding, group_column, speaker_column, include_agents=False, include_links=False, significance=0.05, top_n=20):
        """
        Calculates cosine similarity between each speaker in speaker_list and all other terms (exculding speaker tokens) then calculates Pearson correlations between all terms and the groups in the group_column. Those below the significance level are returned as NA.
        Finally it returns the top_n terms with the highest correlation for each group as a dictionary of this form: {group: {term: correlation}}. If top_n is None, it will return all terms as rows and their correlation to each group as columns. Sorting the values in the columns of that dataframe will give the most and least correlated terms to each group.
        This is useful for finding the terms that are most and least associated with each group in the Speaker Landscape, aiding a qualitative analysis of the characteristic language of each group.


        Parameters:
        embedding (str or Word2Vec embedding): The path to the Word2Vec model or the object conaining the model. The processing.Processing().speaker_landscape() will return that object.
        group_column (str): The column name of the group column.
        speaker_column (list): The column name of the speaker column. This must be the column that was introduced to create the speaker landscape. 
        include_agents (bool): Whether to include the agent tokens in the terms calculated. Default is False.
        include_links (bool): Whether to include the links in the terms calculated. Default is False.
        significance (float): The significance level for the Pearson correlation. Default is 0.05. Correlations below this level will be returned as NA.
        top_n (int): The number of top terms to return for each group. Default is 50. If None, all terms will be returned.

        Returns:
        dict: A dictionary of the top_n terms with the highest correlation for each group as a dictionary of this form: {group: {term: correlation}}.
        or 
        pd.DataFrame: A DataFrame of all terms and their correlation to each group. Rows are the terms, columns represent the groups. Cells are the Pearson coefficients, with those below the significance level returned as NA.
        """

        if isinstance(embedding, str):
            model = Word2Vec.load(embedding)
            model = model.wv
        else:
            model = embedding
        
        df = self.data[[speaker_column, group_column]]
        df.drop_duplicates(inplace=True)

        # Get the speaker tokens
        # df = df.unique()
        speaker_tokens = df[speaker_column].tolist()


        if not any([i.startswith('agent_') for i in speaker_tokens]):
            speaker_tokens = ['_'.join(i.split()) for i in speaker_tokens]
            speaker_tokens = ['agent_' + i for i in speaker_tokens]
            # speaker_tokens = [i.lower() for i in speaker_tokens]
        
        # print(speaker_tokens[:20])
        try:
            a = model.key_to_index.keys()
        except:
            model = model.wv
            a = model.key_to_index.keys()

        sim_df = pd.DataFrame(columns=a)

        # Assuming 'model' is already loaded and 'df' and 'speaker_tokens' are defined
        batch_size = 100  # Set the batch size for processing

        # Initialize an empty list to hold batches of DataFrames
        batch_similarities = []
        counter = 0

        # Loop through the DataFrame in batches
        for batch_start in tqdm(range(0, len(df), batch_size), desc='Speakers batches (100 per batch) left to calculate similarity for'):
            batch_end = min(batch_start + batch_size, len(df))
            current_batch_tokens = speaker_tokens[batch_start:batch_end]

            # Initialize a list to hold the similarity results for the current batch
            batch_data = []

            for i, token in enumerate(current_batch_tokens):
                try:
                    similarities = model.most_similar(token, topn=None)
                    batch_data.append(similarities)
                    # print(token)
                except KeyError:
                    batch_data.append([pd.NA] * len(model.key_to_index))
                    counter += 1

            # Convert the current batch data to a DataFrame
            batch_df = pd.DataFrame(batch_data, columns=model.key_to_index.keys())

            # Append the batch DataFrame to the list of batch similarities
            batch_similarities.append(batch_df)
        print(f'{counter} speakers were not found in the Word2Vec model. They will be returned as NA.')

        # Concatenate all batch DataFrames into the final DataFrame
        sim_df = pd.concat(batch_similarities, ignore_index=True)



        sim_df['group_labels'] = df[group_column]
        sim_df['group_labels'] = sim_df['group_labels'].fillna('NA')
        df_dummies = pd.get_dummies(sim_df, columns=['group_labels'], drop_first=False)

        columns_to_drop = []
        if include_agents==False:
            drop = [col for col in df_dummies.columns if col.startswith('agent_')]
            columns_to_drop += drop
        if include_links==False:
            drop = [col for col in df_dummies.columns if col.startswith('http')]
            columns_to_drop += drop

        df_dummies.drop(columns=columns_to_drop, inplace=True)

        dummy_variables = [col for col in df_dummies.columns if col.startswith('group_labels_')]
        tokens = [col for col in df_dummies.columns if col not in dummy_variables]
        df_dummies = df_dummies.dropna(how='all', subset=tokens)

        correlations = pd.DataFrame(index=tokens, columns=dummy_variables)
        for num_col in tqdm(tokens, desc='Calculating correlations. Tokens left'):
            for dummy_col in dummy_variables:
                corr, p_value = pearsonr(df_dummies[num_col], df_dummies[dummy_col])
                if p_value < significance:
                    correlations.loc[num_col, dummy_col] = corr
        

        if top_n is None:
            return correlations
        else:
            top_terms = {}
            for col in correlations.columns:
                d = correlations[col].dropna()
                terms = d.sort_values(ascending=False)[:top_n]
                terms = {i: terms[i] for i in terms.index}
                top_terms[col] = terms

            return top_terms
        




        




