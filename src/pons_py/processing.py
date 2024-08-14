from fa2_modified import ForceAtlas2
import networkx as nx
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from random import seed, shuffle
import umap.umap_ as umap
import multiprocessing
import string
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
from datetime import datetime
import pandas as pd
import numpy as np

class Processing:
    def __init__(self, data=None):
        try:
            self.data = data.copy()
        except:
            pass

    def social_network(self, edge_list=None, graph=None, weights=None, directed=False, iters=20000, fa2_args=None):
        """
        Function to create a social network from an edge list or NetworkX graph object. It uses the ForceAtlas2 algorithm to create the layout of the network. The function returns the NetworkX graph object and a dataframe with the x and y coordinates of the nodes.
        Note: For the moment, ForceAtlas2 in Python does not support multithreading. If you are working with a large network, it might take a while to compute the layout — if it is too much, we recommend you to use Gephi to compute the layout. You can then export the node table with the x and y positions as a csv, which you can then import into pons_py. 
        Additionally, if you want to further tweak your graph, generate the graph object ouside the function and pass it as an argument, using this only to compute the layout and get the node positions table.


        Args:
        edge_list (pd.DataFrame): Dataframe with the edge list. It should have a 'Source' and 'Target' column. If None, the function will use the data that was passed to the class upon initialisation.
        graph (nx.Graph): NetworkX graph object. If None, the function will use the edge list to create the graph.
        weights (str): Name of the column in the edge list that contains the weights of the edges. If None, the function will assume that all edges have a weight of 1.
        directed (bool): Whether the graph is directed or not. If True, the function will create a directed graph. If False, the function will create an undirected graph.
        iters (int): Number of iterations for the ForceAtlas2 algorithm. Default is 20000.
        fa2_args (dict): Dictionary with arguments to pass to the ForceAtlas2 algorithm. If None, the function will use the default arguments. The default arguments are:
            - outboundAttractionDistribution=True
            - edgeWeightInfluence=1.0
            - jitterTolerance=1.0
            - barnesHutOptimize=True
            - barnesHutTheta=1.2
            - scalingRatio=2.0
            - strongGravityMode=False
            - gravity=1.0
            - verbose=True
        
        Returns:
        g (nx.Graph or nx.DiGraph): NetworkX graph object. It contains the positions as node attributes.
        node_df (pd.DataFrame): Dataframe with the x and y coordinates of the nodes. It also contains a column with the name of the nodes.
        

        """



        fa2 = ForceAtlas2(
                        # Behavior alternatives
                        outboundAttractionDistribution=True,  # Dissuade hubs
                        edgeWeightInfluence=1.0,

                        # Performance
                        jitterTolerance=1.0,  # Tolerance
                        barnesHutOptimize=True,
                        barnesHutTheta=1.2,
                        # multiThreaded=True,  # NOT SUPPORTED

                        # Tuning
                        scalingRatio=2.0,
                        strongGravityMode=False,
                        gravity=1.0,

                        # Log
                        verbose=True)


        if directed:
            directed = nx.Graph()
        else:
            directed = nx.DiGraph()
        
        if edge_list is not None: 
            g = nx.from_pandas_edgelist(edge_list, source='Source', target='Target', edge_attr=weights, create_using=directed)
        elif self.data is not None:
            g = nx.from_pandas_edgelist(self.data, source='Source', target='Target', edge_attr=weights, create_using=directed)
        else:
            g = graph
        
        if fa2_args is not None:
            fa2 = ForceAtlas2(**fa2_args)

        positions = fa2.forceatlas2_networkx_layout(g, pos=None, iterations=iters, weight_attr=weights)
        nx.set_node_attributes(g, positions, 'positions')
    
        node_df = pd.DataFrame(positions).T
        node_df.columns = ['x', 'y']
        node_df['speaker'] = list(g.nodes)
        return g, node_df

    def speaker_landscape(self, data=None, speaker_col='speaker', speech_col='text', preprocess=False, word2vec_args=None, umap_args=None, retain_threshold=5):
        """
        Function to create Speaker Landscapes from text data. It includes preprocessing, training a Word2Vec model, and reducing the dimensionality of the embeddings to 2 and 1 dimensions. During the process it will save the Word2Vec model and the preprocessed text data, for reproducibility and as backup.

        Args:
            data (pd.DataFrame): Dataframe with text data. If None, the function will use the data that was passed to the class upon initialisation.
            speaker_col (str): Name of the column in the dataframe that contains the speaker information.
            speech_col (str): Name of the column in the dataframe that contains the text data.
            preprocess (bool): Whether to preprocess the text data. If True, the function will preprocess the text data by removing punctuation, making all text lowercase, and creating bigrams and ngrams.
            word2vec_args (dict): Dictionary with arguments to pass to the Word2Vec model. If None, the function will use the default arguments. This is 250 dimensions, a window size of 10, skip-gram as training algorithm, 15 epochs, and a minimum count of 5.
            umap_args (dict): Dictionary with arguments to pass to the UMAP model. If None, the function will use the default arguments. This is cosine as the metric, a minimum distance of 0.01, and 40 neighbors.
            retain_threshold (int): Minimum number of text interactions an agent must have to be included in the Speaker Landscape.

        Returns:
            model (Word2Vec): Trained Word2Vec model.
            df (pd.DataFrame): Dataframe with the speaker information, the 2D and 1D coordinates of the speakers, and the number of text interactions of the speakers.
            It also saves the Word2Vec model and the preprocessed text data in the current working directory, both with a timestamp as name.
        

        """

        if data is None:
            df = self.data.copy()
        else:
            df = data

        class DataGenerator(object):
            def __init__(self, path_to_data,
                        share_data=1.,
                        chunk_size=10000,
                        random_buffer_size=100000,
                        data_seed=42):

                """Iterator that loads lines from a (possibly large) file in a mildly randomised fashion.

                A buffer stores a set of lines from the text file. The buffer is shuffled, and the first chunk
                of lines is returned (that is, one such line is yielded each time the generator is called). The buffer
                is filled up again with fresh lines and shuffled. This continues until no lines are left to fill the
                buffer with, at which point the remaining lines are returned.

                Args:
                    path_to_data (str): Full path to a data file with one preprocessed sentence/document per line.
                    share_data (float): How much data to subsample - useful for bootstrapping.
                    chunk_size (int): Return so many lines from the random buffer at once before filling it up again. Larger
                        chunk sizes speed up training, but decrease randomness.
                    random_buffer_size (int): Keep so many lines from the data file in a buffer which is shuffled before
                        returning the samples in a chunk. Higher values take more RAM but lead to more randomness
                        when sampling the data. A value equal to the number of all samples would lead to perfectly
                        random samples.
                    data_seed (int): Random seed for data loading.
                """
                if chunk_size > random_buffer_size:
                    raise ValueError("Chunk size cannot be larger than the buffer size.")

                self.path_to_data = path_to_data
                self.share_of_original_data = share_data
                self.chunk_size = chunk_size
                self.random_buffer_size = random_buffer_size
                seed(data_seed)

            def __iter__(self):

                # load initial buffer
                buffer = []
                with open(self.path_to_data, "r") as f:

                    reached_end = False

                    # fill buffer for the first time
                    for i in range(self.random_buffer_size):
                        line = f.readline().strip().split(" ")
                        if not line:
                            reached_end = True
                            break
                        buffer.append(line)

                    while not reached_end:

                        # randomise the buffer
                        shuffle(buffer)

                        # remove and return chunk from buffer
                        for i in range(self.chunk_size):
                            # separate non-bootstrap case here for speed
                            if self.share_of_original_data == 1.0:
                                yield buffer.pop(0)
                            else:
                                # randomly decide whether this line is in
                                # the bootstrapped data
                                if np.random.rand() > self.share_of_original_data:
                                    buffer.pop(0)
                                    continue
                                else:
                                    yield buffer.pop(0)

                        # fill up the buffer with a fresh chunk
                        for i in range(self.chunk_size):
                            line = f.readline()
                            if not line:
                                reached_end = True
                                break
                            else:
                                buffer.append(line.strip().split(" "))

                    # if end of file has been reached
                    # yield all elements left in the buffer
                    # in random order
                    shuffle(buffer)
                    for el in buffer:
                        yield el

        class PrintLoss(CallbackAny2Vec):
            """Callback to print loss after each epoch."""

            def __init__(self):
                self.epoch = 1
                self.loss_to_be_subed = 0
                self.log = ""

            def on_epoch_end(self, model):
                loss = model.get_latest_training_loss()
                loss_now = loss - self.loss_to_be_subed
                self.loss_to_be_subed = loss
                print('Loss after epoch {}: {}'.format(self.epoch, loss_now))
                self.log += 'Loss after epoch {}: {} \n'.format(self.epoch, loss_now)
                self.epoch += 1



        if preprocess:

            print("Preprocessing data...")
            # We add some symbols to standard punctuation, because they have different ascii representation
            PUNCTUATION = string.punctuation + "“”’‘‚…–—"

            # We remove # and @ signs to preserve them in tweets. Also keep & signs which signify a meaningful word
            PUNCTUATION = PUNCTUATION.replace("#", "").replace("@", "").replace("_", "").replace("&", "")

            def clean_text(text):

                # there seems to be a decoding issue and (some) & are recorded as &amp;. Since
                # this is an important word for meaning, we rectify this here manually
                text = text.replace("&amp;", "&")

                # remove punctuation
                text = ''.join(char for word in text for char in word if char not in PUNCTUATION)

                # make lower case
                text = " ".join([word.lower() for word in text.split()])

                return text.strip("\n").strip(" ")


            df[speech_col] = df[speech_col].apply(clean_text)
            
            def sentence_generator(column):
                for line in df[column]:
                    yield line.strip().split()

            print("Making bigrams...")

            # make the model that builds bigrams
            gram_model = Phrases(sentence_generator(speech_col),
                                min_count=70,
                                threshold=5,
                                max_vocab_size=2000000,
                                connector_words=ENGLISH_CONNECTOR_WORDS)
            gram_model.freeze()


            df[speech_col] = df[speech_col].map(lambda x: " ".join(gram_model[x.strip().split()]))

            print("Making ngrams...")


            # repeat procedure on processed text to get bigrams of words (including bigrams of bigrams)
            gram_model = Phrases(sentence_generator(speech_col),
                                min_count=70,
                                threshold=10,
                                max_vocab_size=2000000,
                                connector_words=ENGLISH_CONNECTOR_WORDS)

            gram_model.freeze()

            df[speech_col] = df[speech_col].map(lambda x: " ".join(gram_model[x.strip().split()]))

            # add agent tokens to the beginning of each sentence

            df[speaker_col] = df[speaker_col].apply(lambda x: "agent_" + "_".join(x.split()))
            
            final_ls = []
            for i, row in df.iterrows():
                line = f'{row[speaker_col]} {row[speech_col]}'
                final_ls.append(line)


            name = datetime.now().strftime("%Y%m%d-%H%M%S")

            with open(f'{name}.txt', 'w') as f:
                for item in final_ls:
                    f.write("%s\n" % item)

        else:
            print("No preprocessing detected. Assuming data is already preprocessed.")
            final_ls = []
            for i, row in df.iterrows():
                line = f'agent_{"_".join(row[speaker_col].split())} {row[speech_col]}'
                final_ls.append(line)
            
            name = datetime.now().strftime("%Y%m%d-%H%M%S")

            with open(f'{name}.txt', 'w') as f:
                for item in final_ls:
                    f.write("%s\n" % item)
        
        training_generator = DataGenerator(f'{name}.txt')

        print("Training word2vec model...")
        if word2vec_args is not None:
            model = Word2Vec(training_generator, **word2vec_args)
        else:
            model = Word2Vec(
                            sentences = training_generator,
                            vector_size = 250, # number of dimensions that the word vectors will have
                            window = 10,  # maximum distance between the current and predicted word
                            sg = 1,  # use skip-gram (semantic learning) as training algorithm
                            workers = multiprocessing.cpu_count(),  # number of threads for training the model
                            min_count = 5,  # ignores all words with total frequency lower than this
                            sorted_vocab = 1, # sort the words in the resulting embedding
                            seed = 42, # use a random seed for reproducability
                            epochs = 15,  # number of times training goes through the data
                            compute_loss = True, # print the loss in each epoch
                            callbacks = [PrintLoss()])
            
        # normalise the word vectors
        model.wv.init_sims()
        model.save(f'{name}.model')

        print('Training done!')

        

        df = df.groupby([speaker_col], as_index=False).agg(text=(speech_col, ' '.join), count=(speech_col, 'count'))
        print("Number of agents in training set: ", len(df.index))

        # only take agents with more than so many tweets
        df = df[df['count'] > retain_threshold]
        print("Number of agents with more than " + str(retain_threshold) + " tweets: ", len(df.index))


        speakers = df[speaker_col].tolist()
        vecs = np.array([model.wv[author] for author in speakers])

        
        

        if umap_args is not None:
            reducer_2d = umap.UMAP(**umap_args)
            reducer_1d = umap.UMAP(n_components=1, **umap_args)
            print('Reducing to 2d')
            vecs_2d = reducer_2d.fit_transform(vecs)
            print('Reducing to 1d')
            vecs_1d = reducer_1d.fit_transform(vecs)
        
        else:
            reducer_2d = umap.UMAP(metric="cosine", min_dist=0.01, n_neighbors=40)
            reducer_1d = umap.UMAP(n_components=1, metric="cosine", min_dist=0.01, n_neighbors=40)
            print('Reducing to 2d')
            vecs_2d = reducer_2d.fit_transform(vecs)
            print('Reducing to 1d')
            vecs_1d = reducer_1d.fit_transform(vecs)
        
        # df["speaker"] = df[speaker_col].map(lambda s: s)
        df[speaker_col] = df[speaker_col].map(lambda s: s[6:])
        df["2d_vec_x"] = vecs_2d[:,0]
        df["2d_vec_y"] = vecs_2d[:,1]
        df["1d_vec"] = vecs_1d

        # df[speaker_col] = df['speaker'].apply(lambda x: x.split('_')[1:])
        # df['speaker'] = df['speaker'].apply(lambda x: ' '.join(x))

        return model, df

    def drop_outliers(self, data, x_col='x', y_col='y', threshold=3):
        """
        Function to drop outliers from a dataframe based on the z-scores of the x and y columns.

        Args:
            data (pd.DataFrame): Dataframe with the data.
            x_col (str): Name of the column in the dataframe that contains the x coordinates.
            y_col (str): Name of the column in the dataframe that contains the y coordinates.
            threshold (int): Threshold for the z-score. If the z-score of a point is higher than this threshold, the function will drop the point.

        Returns:
            data (pd.DataFrame): Dataframe without the outliers.
        """

        data['z_x'] = np.abs((data[x_col] - data[x_col].mean()) / data[x_col].std())
        data['z_y'] = np.abs((data[y_col] - data[y_col].mean()) / data[y_col].std())
        data = data[(data['z_x'] < threshold) & (data['z_y'] < threshold)]
        data = data.drop(columns=['z_x', 'z_y'])
        return data