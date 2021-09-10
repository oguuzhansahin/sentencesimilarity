from sentencesimilarity import sentences_similarity


def test_sentence_similarity():
    model_name = "dbmdz/bert-base-turkish-cased"

    sentences = ['Aynı cümleyi yazıyorum',
                 'Aynı cümleyi yazıyorum']

    sentence_sim = sentences_similarity.SentenceSimilarity(model_name)
    assert sentence_sim.return_most_similar(sentences) == sentences[1]

