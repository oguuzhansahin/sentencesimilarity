from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import math

class SentenceSimilarity():
    
    
    def __init__(self,model_name):
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
    def calculate_similarity(self,sentences):
        '''
        

        Parameters
        ----------
        sentences : List
            Takes a list as an input. The list must be contains sentences.

        Returns
        -------
        Similarity of sentence that is in index 0 with the other sentences.

        '''
        assert isinstance(sentences, list) == True, "You need to pass a list to the function"
        assert len(sentences) > 1, "You must give at least 2 sentences as input"
        mean_pooled = self._create_mean_pooled(sentences)
        
        similarity_result = cosine_similarity(
           [mean_pooled[0]],
           mean_pooled[1:])
           
        print(f"Similarity of the sentence >>> {sentences[0]} <<<< with other sentences:")
        for sent,score in zip(sentences[1:],similarity_result.flatten()):
            print("{:<8} {:<10}".format(sent,score))
     
    
    def return_most_similar(self,sentences):
        mean_pooled = self._create_mean_pooled(sentences)
        
        similarity_result = cosine_similarity(
           [mean_pooled[0]],
           mean_pooled[1:])
        
        most_similar = sentences[similarity_result.flatten().tolist().index(max(similarity_result.flatten()))+1]
        return most_similar
        
    def _create_mean_pooled(self,sentences):
      
       
      
       tokens = {'input_ids': [],
                 'attention_mask': []}
       
       max_len = max([len(self.tokenizer.tokenize(sentence)) for sentence in sentences])
       if max_len > 512:
           print("Your sentence length more than 512. That's why it will be truncated to 512.")
           max_len = 512
           
       for sentence in sentences:
           
           n_tokens = self.tokenizer.encode_plus(sentence,
                                                 max_length = max_len,
                                                 truncation=True,
                                                 padding = 'max_length',
                                                 return_tensors="pt")
           
           tokens['input_ids'].append(n_tokens['input_ids'][0])
           tokens['attention_mask'].append(n_tokens['attention_mask'][0])

       tokens['input_ids'] = torch.stack(tokens['input_ids'])
       tokens['attention_mask'] = torch.stack(tokens['attention_mask'])
       
       logits = self.model(**tokens)
       
       embeddings = logits.last_hidden_state
       mask = tokens['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
       
       masked_embeddings = embeddings * mask
       
       summed = torch.sum(masked_embeddings,axis=1)
       summed_mask = torch.clamp(mask.sum(1), min=1e-9)
       
       mean_pooled = summed/summed_mask
       mean_pooled = mean_pooled.detach().numpy()
       
       return mean_pooled
#%%



