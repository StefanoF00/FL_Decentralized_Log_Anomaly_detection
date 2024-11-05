import os
import pandas as pd
import numpy as np
import json
import time
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from sklearn.decomposition import PCA

class Callback(CallbackAny2Vec):
    """
    Callback to print loss after each epoch
    """
    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            print('- Loss after epoch {}: {}'.format(self.epoch+1, loss))
        else:
            print('- Loss after epoch {}: {}'.format(self.epoch+1, loss - self.loss_previous_step))
        self.epoch += 1
        self.loss_previous_step = loss


class LogEmbedder:
    def __init__(self, vector_size=300, window=5, min_count=0, workers=4, alpha=0.01, epochs=10):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.alpha = alpha
        self.epochs = epochs

    def get_sentences(self, path):
        """
        Reads the templates file.

        Args:
        - path: csv file of log templates.

        Returns:
        - labels: list of log keys.
        - templates: list of log templates.
        - sentences: list of templates, where each template is represented as a list of words.
        """
        data = pd.read_csv(path).values
        templates = data[:, 1]
        labels = data[:, 0]
        sentences = [s.split() for s in templates]
        return labels, templates, sentences

    def train_model(self, sentences, model_path):
        """
        Trains the Word2Vec model and saves it.

        Args:
        - sentences: list of sentences for training.
        - model_path: path to save the trained model.
        """
        model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            alpha=self.alpha,
            compute_loss=True,
            callbacks=[Callback()],
            epochs=self.epochs
        )
        model.save(model_path)

    def add_embedding_column(self, input_csv, output_csv, model_path):
        """
        Associates to each template sequence the corresponding embedded sequence.

        Args:
        - input_csv: csv file of log templates.
        - output_csv: output file with embedded sequences.
        - model_path: path to the trained word2vec model.
        """
        model = Word2Vec.load(model_path)
        data = pd.read_csv(input_csv)
        embeddings = []
        for template in data['EventTemplate']:
            embedding = sum(model.wv[word] for word in template.split() if word in model.wv) / len(template.split())
            normalized_embedding = embedding / np.linalg.norm(embedding)
            embedding_json = json.dumps(normalized_embedding.tolist())
            embeddings.append(embedding_json)
        data['Embedding300'] = embeddings
        data.to_csv(output_csv, index=False)

    def cosine_similarity(self, vec1, vec2):
        """
        Computes cosine similarity between two vectors.

        Args:
        - vec1 (numpy.ndarray): first vector.
        - vec2 (numpy.ndarray): second vector.

        Returns:
        - similarity: float value representing cosine similarity.
        """
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0  # Avoid division by zero

        similarity = dot_product / (norm_vec1 * norm_vec2)
        return similarity
    
class EmbeddingReducer:
    def __init__(self, n_components, n_directions):
        self.n_components = n_components
        self.n_directions = n_directions

    def PPA(self, input_path, output_path):
        """
        Algorithm for generation of low dimension semantic vector.

        Args:
        - input_path: .json file containing high dim semantic vectors.
        - n: number of components for PCA dimensionality reduction.
        - d: number of dominating eigen-directions to remove.
        - output_path: .json file in which PPA result will be saved.

        Returns:
        - ppa_result: numpy array of low dimension semantic vectors.
        """
        df = pd.read_csv(input_path)

        # Step 0: open file (structure of dictionary) & find semantic vectors
        high_dim_semantic_vectors = []
        for embedding_json in df['Embedding300']:
            embedding = json.loads(embedding_json)
            high_dim_semantic_vectors.append(embedding)
        print(f'- Logs of this client are classified in {len(high_dim_semantic_vectors)} keys')
        print(f'- Length of each high dim semantic vector: {len(high_dim_semantic_vectors[0])}')

        # Step 1: Semantic_Vectors <- PCA(High_dim_Semantic_Vectors)
        estimator = PCA(n_components=self.n_components)
        Semantic_Vectors = estimator.fit_transform(high_dim_semantic_vectors)

        # Step 2: Semantic_Vectors <- Semantic_Vectors - Average(Semantic_Vectors)
        Semantic_Vectors = Semantic_Vectors - np.mean(Semantic_Vectors, axis=0)

        # Step 3: Semantic_Vectors <- PCA(Semantic_Vectors)
        pca = PCA(n_components=self.n_components)
        Semantic_Vectors = pca.fit_transform(Semantic_Vectors)

        # Step 4: removing d dominating eigenvectors from each semantic vec x in Semantic_Vectors
        U = pca.components_
        ppa_result = []
        for x in Semantic_Vectors:
            for u in U[:self.n_directions]:
                x = x - np.dot(u, x) * u
            ppa_result.append(list(x))
        print(f'- Algorithm terminated: length of each low dim semantic vector: {len(ppa_result[0])}')

        # Save PPA result to CSV
        ppa_result_json = [json.dumps(vec) for vec in ppa_result]
        df["EmbeddingPPA"] = ppa_result_json
        df.to_csv(output_path, index=False)
        ppa_result = np.array(ppa_result)
        return ppa_result


class ClientEmbedder:
    def __init__(self, client_id, parsed_data_dir, embedded_data_dir, w2v_models_dir, log_embedder, ppa_processor):
        self.client_id = client_id
        self.parsed_data_dir = parsed_data_dir
        self.embedded_data_dir = embedded_data_dir
        self.w2v_models_dir = w2v_models_dir
        self.log_embedder = log_embedder
        self.ppa_processor = ppa_processor

    def train_word2vec_model(self):
        print(f'\n* Train word2vec model for client {self.client_id}...')

        # Define the paths
        templates_client_data = f'{self.parsed_data_dir}client_data_{self.client_id}/client_{self.client_id}.log_templates.csv'
        embedded_client_data_dir = f'{self.embedded_data_dir}client_data_{self.client_id}/'
        w2v_model = f'{self.w2v_models_dir}w2v_model_client_{self.client_id}.pth'

        if not os.path.exists(embedded_client_data_dir):
            os.makedirs(embedded_client_data_dir)

        labels, templates, sentences = self.log_embedder.get_sentences(templates_client_data)

        self.log_embedder.train_model(sentences, w2v_model)
        print('- Done!', end=' ')

        print(f'Associating to each template sequence the corresponding embedded sequence for client {self.client_id}...',end='')
        embedded_client_data = f'{embedded_client_data_dir}client_{self.client_id}.log_embedding.csv'
        self.log_embedder.add_embedding_column(templates_client_data, embedded_client_data, w2v_model)
        print('Done!')

        # Perform cosine similarity
        df = pd.read_csv(embedded_client_data)
        vec1 = json.loads(df["Embedding300"][0])
        vec2 = json.loads(df["Embedding300"][0])
        similarity = self.log_embedder.cosine_similarity(vec1, vec2)
        print(f"- Cosine similarity between the first two embeddings: {similarity:.4f}")

    def apply_ppa(self):
        print(f'\n* Applying PPA to client {self.client_id}...')
        # Define the paths
        embedded_client_data_dir = f'{self.embedded_data_dir}client_data_{self.client_id}/'
        embedded_client_data = f'{embedded_client_data_dir}client_{self.client_id}.log_embedding.csv'
        embedded_PPA_client_data = f'{embedded_client_data_dir}client_{self.client_id}.log_embedding_ppa.csv'

        if not os.path.exists(embedded_client_data_dir):
            os.makedirs(embedded_client_data_dir)
        low_dim_semantic_vectors = self.ppa_processor.PPA(embedded_client_data, embedded_PPA_client_data)