from typing import Optional

from gensim.parsing.preprocessing import preprocess_documents
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
from sklearn.decomposition import PCA

from util import load_chapter, load_index


def train_model(path: str, pca: Optional[int], **kwargs) -> np.array:
    """
    Trains a gensim word2vec model on the given input and returns the resulting
    vector representations.

    :param path: The project directory to train a model on the chapters of.
    :param pca: If not None, converts the vectors returned by gensim into a
        `pca`-dimensional representation.
    :param kwargs: Any additional parameters for the model.
    :return: The vector representations of the chapters.
    """
    raw_documents = [load_chapter(path, c) for c in load_index(path)]
    documents = [TaggedDocument(d, [i]) for i, d in
                 enumerate(preprocess_documents(raw_documents))]

    model = Doc2Vec(documents, **kwargs)

    pts = np.array([model.dv[i] for d, [i] in documents])

    if pca is not None:
        pts -= pts.mean(axis=0)
        pts /= pts.std(axis=0)
        pts = PCA(n_components=pca).fit_transform(pts)

    return pts
