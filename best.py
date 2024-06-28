import torch
import torch.nn.functional as F
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve
from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForCausalLM, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer

def euclidean_similarity(v1, v2):
    return 1/(1+euclidean_distances(v1,v2)[0][0])

class TestModel():
    numOfModels = 11
    model_names = [
            "albert-base-v2",
            "sentence-transformers/paraphrase-MiniLM-L12-v2",
            "sentence-transformers/paraphrase-MiniLM-L6-v2",
            "sentence-transformers/paraphrase-MiniLM-L3-v2",
            "MASK FILLER \talbert-base-v2",
            "PPL CALCULATOR \tdistilgpt2",
            "PPL CALCULATOR \tbert-base-uncased",
            "P sentence-transformers/paraphrase-MiniLM-L12-v2",
            "P sentence-transformers/paraphrase-MiniLM-L6-v2",
            "P sentence-transformers/paraphrase-MiniLM-L3-v2",
            "MANUAL MASKS \talbert-base-v2",

    ]
    model_rep_names = [
            "albert-base-v2",
            "sentence-transformers/paraphrase-MiniLM-L12-v2",
            "sentence-transformers/paraphrase-MiniLM-L6-v2",
            "sentence-transformers/paraphrase-MiniLM-L3-v2",
            "albert-base-v2",
            "distilgpt2",
            "bert-base-uncased",
            "sentence-transformers/paraphrase-MiniLM-L12-v2",
            "sentence-transformers/paraphrase-MiniLM-L6-v2",
            "sentence-transformers/paraphrase-MiniLM-L3-v2",
            "albert-base-v2",

    ]
    def __init__(self, model):
        model_names = TestModel.model_names
        model_rep_names = TestModel.model_rep_names
        self.topk = 250
        print("="*50)
        print(f"\nUsing {model_names[model]} of sequence ID {model}:")
        if model == 0:
            self.tokenizer = AutoTokenizer.from_pretrained(model_rep_names[model])
            self.model = AutoModel.from_pretrained(model_rep_names[model])
        elif model == 1:
            self.tokenizer = AutoTokenizer.from_pretrained(model_rep_names[model])
            self.model = AutoModel.from_pretrained(model_rep_names[model])
        elif model == 2:
            self.tokenizer = AutoTokenizer.from_pretrained(model_rep_names[model])
            self.model = AutoModel.from_pretrained(model_rep_names[model])
        elif model == 3:
            self.tokenizer = AutoTokenizer.from_pretrained(model_rep_names[model])
            self.model = AutoModel.from_pretrained(model_rep_names[model])            
        elif model == 4:
            self.tokenizer = AutoTokenizer.from_pretrained(model_rep_names[model])
            self.mask_filler = pipeline('fill-mask', model=model_rep_names[model])
        elif model == 5:
            self.tokenizer = AutoTokenizer.from_pretrained(model_rep_names[model])
            self.model = AutoModelForCausalLM.from_pretrained(model_rep_names[model])
        elif model == 6:
            self.tokenizer = AutoTokenizer.from_pretrained(model_rep_names[model])
            self.model = AutoModelForCausalLM.from_pretrained(model_rep_names[model])
        elif model >= 7 and model <= 9:
            self.model = SentenceTransformer(model_rep_names[model])
        elif model == 10:
            self.tokenizer = AutoTokenizer.from_pretrained(model_rep_names[model])
            self.manual_unmasker = AutoModelForMaskedLM.from_pretrained(model_rep_names[model])
            print("哟西")
        else:
            raise ("Model ID OUT OF RANGE!")

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def find_subarray_index(self, arr, subarr):
        arr_tensor = arr.detach()
        subarr_tensor = subarr.detach()
        n = arr_tensor.size(0)
        m = subarr_tensor.size(0)

        for i in range(n - m + 1):
            if torch.equal(arr_tensor[i:i + m], subarr_tensor):
                return i
        return 'not_found'

    def getems(self, word, sentence):
        word_tokenized = self.tokenizer(word, return_tensors='pt')
        sentence_tokenized = self.tokenizer(sentence, return_tensors='pt')
        insentence_pos = self.find_subarray_index(sentence_tokenized.input_ids.squeeze(), word_tokenized.input_ids.squeeze()[1:-1])
        if insentence_pos == 'not_found':
            print(f"Not Found for '{word}' in '{sentence}'")
            raise Exception("NOT FOUND!")
        sentence_tokenized.attention_mask[0][insentence_pos: insentence_pos+len(word_tokenized.input_ids.squeeze())-2] = sentence_tokenized.attention_mask[0][insentence_pos: insentence_pos+len(word_tokenized.input_ids.squeeze())-2].mul(1)
        #print(sentence_tokenized)

        with torch.no_grad():
            word_out = self.model(**word_tokenized)
            sentence_out = self.model(**sentence_tokenized)

        word_embeddings = word_out.last_hidden_state.squeeze()
        sentence_embeddings = sentence_out.last_hidden_state.squeeze()
        insentence_embeddings = sentence_embeddings[insentence_pos: insentence_pos+len(word_tokenized.input_ids.squeeze())-2]
        insentence_mask = sentence_tokenized.attention_mask.squeeze()[insentence_pos: insentence_pos+len(word_tokenized.input_ids.squeeze())-2]
        insentence_mean_embedding = self.mean_pooling(insentence_embeddings.unsqueeze(0), insentence_mask.unsqueeze(0))
        sentence_mean_embedding = self.mean_pooling(sentence_embeddings.unsqueeze(0), sentence_tokenized.attention_mask)
        word_mean_embedding = self.mean_pooling(word_embeddings[1:-1].unsqueeze(0), word_tokenized.attention_mask.squeeze()[1:-1].unsqueeze(0))

        return word_mean_embedding, insentence_mean_embedding
        # return word_mean_embedding, sentence_mean_embedding

    def mask_fill_comparison(self, original_word, sentence, new_word):
        # Using regex to replace only the first occurrence of the whole word
        pattern = r'\b{}\b'.format(re.escape(original_word))
        masked_sentence = re.sub(pattern, '[MASK]', sentence, count=1)
        
        predictions = self.mask_filler(masked_sentence, top_k=self.topk)
        predicted_words = [pred['token_str'][0:] for pred in predictions]
    
        return 1 if new_word in predicted_words else 0

    def logits_eval(self, original_word, sentence, new_word):
        originalS = sentence.split(original_word, 1)
        originalSplit = [ "[CLS]"+originalS[0], original_word, originalS[1]+"[SEP]" ]
        newSplit = [ "[CLS]"+originalS[0], new_word, originalS[1]+"[SEP]" ]
        #
        originalIdsPart1 = self.tokenizer.encode(originalSplit[0], add_special_tokens=False)
        originalIdsPart2 = self.tokenizer.encode(originalSplit[1], add_special_tokens=False)
        originalIdsPart3 = self.tokenizer.encode(originalSplit[2], add_special_tokens=False)
        originalIds = originalIdsPart1+originalIdsPart2+originalIdsPart3
        originalPart2Poses = [len(originalIdsPart1), len(originalIdsPart1)+len(originalIdsPart2)]
        #
        newIdsPart1 = self.tokenizer.encode(newSplit[0], add_special_tokens=False)
        newIdsPart2 = self.tokenizer.encode(newSplit[1], add_special_tokens=False)
        newIdsPart3 = self.tokenizer.encode(newSplit[2], add_special_tokens=False)
        newIds = newIdsPart1+newIdsPart2+newIdsPart3
        newPart2Poses = [len(newIdsPart1), len(newIdsPart1)+len(newIdsPart2)]
        #
        if len(newIdsPart2)+len(originalIdsPart2)>2:
            print([original_word, sentence, new_word])
        #
        original_input_ids = torch.tensor(originalIds).reshape(1, -1)
        original_attention_mask = torch.ones(len(originalIds)).reshape(1, -1)
        original_attention_mask[0][originalPart2Poses[0] : originalPart2Poses[1]] = original_attention_mask[0][originalPart2Poses[0] : originalPart2Poses[1]].mul(0)
        #original_attention_mask[0][originalPart2Poses[0] : originalPart2Poses[1]] = original_attention_mask[0][originalPart2Poses[0] : originalPart2Poses[1]].mul(1)
        #
        new_input_ids = torch.tensor(newIds).reshape(1, -1)
        new_attention_mask = torch.ones(len(newIds)).reshape(1, -1)
        new_attention_mask[0][newPart2Poses[0] : newPart2Poses[1]] = new_attention_mask[0][newPart2Poses[0] : newPart2Poses[1]].mul(0)
        #new_attention_mask[0][newPart2Poses[0] : newPart2Poses[1]] = new_attention_mask[0][newPart2Poses[0] : newPart2Poses[1]].mul(1)
        #
        original_logit = self.manual_unmasker(input_ids=original_input_ids, attention_mask=original_attention_mask).logits[0]
        new_logit = self.manual_unmasker(input_ids=new_input_ids, attention_mask=new_attention_mask).logits[0]
        #
        original_reasonability = 1
        for idx, tokenPos in enumerate(range(*originalPart2Poses)):
            original_reasonability *= original_logit[tokenPos][originalIdsPart2[idx]]
        original_reasonability **= 1/(idx+1)
        #
        new_reasonability = 1
        for idx, tokenPos in enumerate(range(*newPart2Poses)):
            new_reasonability *= new_logit[tokenPos][newIdsPart2[idx]]
        new_reasonability **= 1/(idx+1)
        #
        return original_reasonability, new_reasonability

    def calculate_perplexity(self, sentence):
        # Tokenize input sentence
        inputs = self.tokenizer(sentence, return_tensors='pt')
        
        # Move tensors to the GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        inputs = {key: value.to(device) for key, value in inputs.items()}
        self.model.to(device)
        
        # Get the loss (negative log likelihood)
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            #print("ouputs:", outputs)
        
        # Calculate perplexity
        perplexity = torch.exp(loss)
        return perplexity.item()

    def interactive_test(self):
        global word1, context1, word1embedding, context1embedding, word2, context2, word2embedding, context2embedding

        word1 = input("word 1: ")
        context1 = input("context 1: ")
        word1embedding, context1embedding = self.getems(word1, context1)

        word2 = input("word 2: ")
        context2 = input("context 2: ")
        word2embedding, context2embedding = self.getems(word2, context2)

        #print("\nComparison(cosine): ", cosine_similarity(context1embedding, context2embedding))
        #print("\nComparison(euclidean): ", euclidean_similarity(context1embedding, context2embedding))

    def cald(self, sentence, new_sentence, original_word, new_word):
        _, originalem = self.getems(original_word, sentence)
        _, tryem = self.getems(new_word, new_sentence)
        #print("\nComparison(cosine): ", cosine_similarity(originalem, tryem))

if __name__ == '__main__':
    sentences_pairs = [
        ('bank', 'We sat on the bank of the river, watching the boats go by.', 'shore', 'We sat on the shore of the river, watching the boats go by.', 1),
        ('bank', 'The bank offered a new type of savings account with a higher interest rate.', 'shore', 'The shore offered a new type of savings account with a higher interest rate.', 0),
        ('bank', 'The ducks waddled along the bank, searching for food.', 'shore', 'The ducks waddled along the shore, searching for food.', 1),
        ('bank', 'After cashing my paycheck, I deposited the money in my bank account.', 'shore', 'After cashing my paycheck, I deposited the money in my shore account.', 0),
        ('bare', 'The family was short of even the bare necessities of life.', 'naked', 'The family was short of even the naked necessities of life.', 0),
        ('bare', 'She likes to walk around in bare feet.', 'naked', 'She likes to walk around in naked feet.', 1),
        ('sentence', 'A sentence is a group of words which, when they are written down, begin with a capital letter and end with a full stop, question mark, or exclamation mark. Most sentences contain a subject and a verb.', 'conviction', 'A conviction is a group of words which, when they are written down, begin with a capital letter and end with a full stop, question mark, or exclamation mark. Most sentences contain a subject and a verb.', 0),
        ('sentence', 'They are already serving prison sentence for their part in the assassination.', 'conviction', 'They are already serving prison conviction for their part in the assassination.', 1),
        ('lead', 'She followed the lead.', 'metal', 'She followed the metal.', 0)
    ]
    sentences_pairs.extend([
        ('light', 'She turned on the light to read her book.', 'lamp', 'She turned on the lamp to read her book.', 1),
        ('light', 'He prefers to travel light, carrying only a backpack.', 'lamp', 'He prefers to travel lamp, carrying only a backpack.', 0),
        ('rock', 'The band played rock music all night long.', 'stone', 'The band played stone music all night long.', 0),
        ('rock', 'They climbed the steep rock.', 'stone', 'They climbed the steep stone.', 1),
        ('ring', 'The phone began to ring.', 'circle', 'The phone began to circle.', 0),
        ('current', 'The current in the river was very strong.', 'flow', 'The flow in the river was very strong.', 1),
        ('current', 'She is very up-to-date with current events.', 'flow', 'She is very up-to-date with flow events.', 0),
        ('seal', 'She watched the seal swim gracefully in the zoo.', 'stamp', 'She watched the stamp swim gracefully in the zoo.', 0),
        ('seal', 'He placed his seal on the letter for authenticity.', 'stamp', 'He placed his stamp on the letter for authenticity.', 1),
        ('bark', 'The dog began to bark loudly.', 'tree', 'The dog began to tree loudly.', 0),
        ('bark', 'She noticed the rough bark of the tree.', 'tree', 'She noticed the rough tree of the tree.', 0),
        ('season', 'He likes to season his food with various spices.', 'period', 'He likes to period his food with various spices.', 0),
        ('book', 'I need to book a hotel room for my trip.', 'reserve', 'I need to reserve a hotel room for my trip.', 1),
        ('book', 'I am currently reading a great book.', 'reserve', 'I am currently reading a great reserve.', 0),
    ])

    sentences_pairs.extend([
        ('match', 'They watched the football match together.', 'game', 'They watched the football game together.', 1),
        ('match', 'She lit the match to start the fire.', 'game', 'She lit the game to start the fire.', 0),
        ('star', 'The actor became a huge star.', 'celebrity', 'The actor became a huge celebrity.', 1),
        ('star', 'They watched the star twinkle in the night sky.', 'celebrity', 'They watched the celebrity twinkle in the night sky.', 0),
        ('wave', 'She gave him a cheerful wave as he left.', 'gesture', 'She gave him a cheerful gesture as he left.', 1),
        ('wave', 'The wave knocked them off their feet.', 'gesture', 'The gesture knocked them off their feet.', 0),
        ('kind', 'She is very kind to animals.', 'type', 'She is very type to animals.', 0),
        ('kind', 'What kind of music do you like?', 'type', 'What type of music do you like?', 1),
        ('train', 'He caught the morning train to work.', 'car', 'He caught the morning car to work.', 0),
        ('train', 'She decided to train for a marathon.', 'prepare', 'She decided to prepare for a marathon.', 1),
        ('plate', 'The tectonic plate shifted during the earthquake.', 'dish', 'The tectonic dish shifted during the earthquake.', 0),
        ('point', 'She made a valid point during the discussion.', 'argument', 'She made a valid argument during the discussion.', 1),
        ('point', 'He used a laser pointer to highlight his point on the screen.', 'argument', 'He used a laser pointer to highlight his argument on the screen.', 1),
        ('pound', 'He lost a pound after his workout.', 'weight', 'He lost a weight after his workout.', 0),
        ('pound', 'She had to pound the stakes into the ground.', 'hammer', 'She had to hammer the stakes into the ground.', 1),
        ('right', 'Turn to the right please.', 'correct', 'Turn to the correct please.', 0),
    ])

    sentences_pairs.extend([
        ('fellow', 'My fellow citizens, for the last nine days, the entire world has seen for itself the state of our Union -- and it is strong.', 'companion', 'My companion citizens, for the last nine days, the entire world has seen for itself the state of our Union -- and it is strong.', 0),
        ('but', 'Americans have known the casualties of war -- but not at the center of a great city on a tranquil morning.', 'merely', 'Americans have known the casualties of war -- merely not at the center of a great city on a tranquil morning.', 0)
    ]);

    model_scores = []
    predicted_classeses = []

    for i in range(TestModel.numOfModels):
    #for i in [10]:
        model_name = TestModel.model_names[i]
        testModelInstance = TestModel(i)
        cosine_similarities = []

        for word1, sentence1, word2, sentence2, ground_truth in sentences_pairs:
            if i == 4:
                valid_replacement = testModelInstance.mask_fill_comparison(word1, sentence1, word2)
                cosine_similarities.append((valid_replacement, ground_truth))
                #print(f"\nUsing {model_name}:")
                #print(f"Comparison of [{word1}] in [{sentence1}] and [{word2}] in [{sentence2}]")
                #print("Mask-Filling Result: ", valid_replacement)
            elif i >= 5 and i<7:
                # Calculate perplexity for both original and new sentences
                perplexity1 = testModelInstance.calculate_perplexity(sentence1)
                perplexity2 = testModelInstance.calculate_perplexity(sentence2)
                perplexity_simi = perplexity1 - perplexity2
                cosine_similarities.append((perplexity_simi, ground_truth))

                #print(f"\nUsing {model_name}:")
                #print(f"Comparison of [{word1}] in [{sentence1}] and [{word2}] in [{sentence2}]")
                #print("Perplexity Difference: ", perplexity_diff)
            elif i <= 3:
                wembed1, embedding1 = testModelInstance.getems(word1, sentence1)
                wembed2, embedding2 = testModelInstance.getems(word2, sentence2)
                cos_sim = cosine_similarity(  wembed1-embedding1, wembed1-embedding2 )[0][0]
                #cos_sim = cosine_similarity( wembed2, embedding2 )[0][0]
                #cos_sim = cosine_similarity(  wembed2-embedding1, wembed2-embedding2 )[0][0]
                #cos_sim = cosine_similarity(  wembed1-embedding1, wembed2-embedding1 )[0][0]
                #cos_sim = cosine_similarity(  wembed1-embedding2, wembed2-embedding2 )[0][0]
                # cos_sim = cosine_similarity( embedding1, embedding2 )[0][0] # L6 F1: 0.69
                cosine_similarities.append((cos_sim, ground_truth))

                #print(f"\nUsing {model_name}:")
                #print(f"Comparison of [{word1}] in [{sentence1}] and [{word2}] in [{sentence2}]")
                #print("Comparison(cosine): ", cos_sim)
                #print("Comparison(euclidean): ", euclidean_similarity(embedding1, embedding2))
            elif i >= 7 and i <= 9:
                embeddings = testModelInstance.model.encode([sentence1, sentence2])
                cos_sim = cosine_similarity(*embeddings.reshape(2, 1, -1))[0][0]
                cosine_similarities.append((cos_sim, ground_truth))
            elif i == 10:
                originalReasonability, newReasonability = testModelInstance.logits_eval(word1, sentence1, word2)
                #cosim = newReasonability.item()-originalReasonability.item()
                cosim = newReasonability.item()
                cosine_similarities.append((cosim, ground_truth))
            else:
                raise 'DAMN'


        model_cos_sim, ground_truths = zip(*cosine_similarities)
        
        # Determine the best threshold using ROC curve
        fpr, tpr, thresholds = roc_curve(ground_truths, model_cos_sim)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        print("Threshold for ", model_name, " is ", optimal_threshold)

        # Classify using the optimal threshold
        predicted_classes = [1 if sim >= optimal_threshold else 0 for sim in model_cos_sim]

        accuracy = accuracy_score(ground_truths, predicted_classes)
        precision = precision_score(ground_truths, predicted_classes)
        recall = recall_score(ground_truths, predicted_classes)
        f1 = f1_score(ground_truths, predicted_classes)

        predicted_classeses.append(model_cos_sim)
        model_scores.append((model_name, accuracy, precision, recall, f1))

    # Rank models by their F1 score
    zipped = zip(predicted_classeses, model_scores)
    ranked_zip = sorted(zipped, key=lambda ze: ze[1][4], reverse=True)
    ranked_classes, ranked_scores = zip(*ranked_zip)

    print("\n\nModel Rankings based on F1 Score:\n")
    for rank, (classes, (model_name, accuracy, precision, recall, f1)) in enumerate(ranked_zip, 1):
        print(f"{rank}. {model_name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
        print(classes)
        print("="*50)
