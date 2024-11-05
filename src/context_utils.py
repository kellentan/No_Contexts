#################################################
############### Context Utilities ###############
#################################################
# Import packages
import json
import torch
import pickle
import random
import typing
import datasets
import evaluate
import fuzzywuzzy
import numpy as np
import transformers
import pandas as pd
from tqdm.auto import tqdm
from fuzzywuzzy import fuzz

# Global parameters
ACCURACY = evaluate.load("/scratch/gpfs/kc4642/Metrics/accuracy.py")
with open("IMPLI_data/impli_df.pkl", "rb") as file: IMPLI_DF = pickle.load(file)

if (torch.cuda.is_available()): DEVICE = torch.device("cuda:0")
elif (torch.backends.mps.is_available()): DEVICE = torch.device("mps")
else: DEVICE = torch.device("cpu")

# Setup methods
def set_logging_and_seed(seed: int=42) -> None:
    '''
    A helper method to help set all the random seeds when copied into each notebook cell.

    inputs:
        seed -> (int, optional) the number to use for the random seed
    outputs:
        None
    '''
    # Logging output settings
    transformers.utils.logging.set_verbosity_info()
    logger = transformers.utils.logging.get_logger("transformers")
    transformers.utils.logging.set_verbosity(30)
    logger.warning("WARN")

    # Random seet outputs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if (torch.cuda.is_available()): torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)
    return

def load_model(model_path: str, tokenizer_path: str, device: torch.device=DEVICE, freeze: bool=False, verbose: bool=False) -> typing.Tuple[typing.Union[transformers.AutoModelForSequenceClassification, None], typing.Union[transformers.AutoTokenizer, None]]:
    '''
    Given the model and tokenizer paths, load the inference model and tokenizer.

    inputs:
        model_path     -> (str) the local path for the classification transformer model
        tokenizer_path -> (str) the local path for the tokenizer
        device         -> (torch.device, optional) the device to host our model
        freeze         -> (bool, optional) if set, will freeze all parameters in the base model
        verbose        -> (bool, optional) if set, print out any indicator information
    outputs:
        model          -> (transformers.AutoModelForSequenceClassification) the classification transformer model
        tokenizer      -> (transformers.AutoTokenizer) the tokenizer
    '''
    # Try loading with the given model and tokenizer path
    try:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path,
                                                                                problem_type="multi_label_classification",
                                                                                num_labels=2,
                                                                                local_files_only=True,
                                                                                ignore_mismatched_sizes=True)
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)

        # Add explicit pad token if one doesn't exist
        if (tokenizer.pad_token is None):
            tokenizer.add_special_tokens({"pad_token" : "<pad>"})
            model.resize_token_embeddings(len(tokenizer))
            model.config.pad_token_id = tokenizer.pad_token_id
            if (verbose == True): print("We have manually configured the pad tokens.")

        # For non-BART models, assume left padding (implicitly assumes decoder-only LLMs for non-BART models)
        if ("bart" not in model_path): tokenizer.padding_side = "left"

        # Freeze all parameters in the base model if set
        if (freeze == True):
            for parameter in model.model.parameters(): parameter.requires_grad = False
            if (verbose == True): print("We have frozen all parameters in the base model.")
        pass
    except:
        print("Error: you used either an invalid model or tokenizer path.")
        print(f"Model Path: {model_path} | Tokenizer Path: {tokenizer_path}")
        model, tokenizer = None, None

    return model, tokenizer

# Data preprocessing methods
def label2onehot(entailment: list, nonentailment: list, adversarial: list, construct: bool=False) -> typing.Tuple[list, list, list]:
    '''
    Given the NLI text samples, create onehot labels for classification.

    inputs:
        entailment    -> (list) a list of all entailment text samples
        nonentailment -> (list) a list of all nonentailment text samples
        adversarial   -> (list) a list of all adversarial text samples
        construct     -> (bool, optional) if set, will overwrite the current sample entirely
    outputs:
        entailment    -> (list) a list of the final entailment text samples
        nonentailment -> (list) a list of the final nonentailment text samples
        adversarial   -> (list) a list of the final adversarial text samples
    '''
    if (construct == True):
        for i in range(len(entailment)): entailment[i] = [entailment[i][0], entailment[i][1], [1.0, 0.0]]
        for i in range(len(nonentailment)): nonentailment[i] = [nonentailment[i][0], nonentailment[i][1], [0.0, 1.0]]
        for i in range(len(adversarial)): adversarial[i] = [adversarial[i][0], adversarial[i][1], [0.0, 1.0]]
    else:
        for i in range(len(entailment)): entailment[i][2] = [1.0, 0.0]
        for i in range(len(nonentailment)): nonentailment[i][2] = [0.0, 1.0]
        for i in range(len(adversarial)): adversarial[i][2] = [0.0, 1.0]
         
    return entailment, nonentailment, adversarial

def length_filtering(dataset: list, min_length: int=6, max_length: int=64, invalid_labels: list=[], verbose: bool=False) -> list:
    '''
    Given a list of NLI samples, exclude all samples with number of words exceeding the maximum or below the minimum count.

    inputs:
        dataset        -> (list) a list of text samples
        min_length     -> (int, optional) the minimum number of words in each sample
        max_length     -> (int, optional) the maximum number of words in each sample
        invalid_labels -> (list, optional) the list of invalid labels from which to filter samples
        verbose        -> (bool, optional) if set, print out indicator information
    outputs:
        dataset        -> (list) the filtered list of valid text samples
    '''
    premises, hypotheses = [], []

    # Identify invalid data samples
    for i in range(len(dataset)):
        premise, hypothesis, label = dataset[i][0], dataset[i][1], dataset[i][2]
        premise_length, hypothesis_length = len(premise.split(" ")), len(hypothesis.split(" "))

        # Note the index for an invalid premise
        if (premise_length >= max_length or (min_length != None and premise_length <= min_length)): premises.append(i)

        # Note the index for an invalid hypothesis
        if (hypothesis_length >= max_length or (min_length != None and hypothesis_length <= min_length)): hypotheses.append(i)

        # Note the index for an invalid label
        if (label in invalid_labels): premises.append(i)
        pass

    # Delete all samples that are invalid
    invalid_idx = list(set(premises + hypotheses))
    for idx in sorted(invalid_idx, reverse=True): del dataset[idx]
    if (verbose == True): print(f"we found {len(invalid_idx)} invalid indices.")

    return dataset

def load_data(dataset_dir: str, clean_labels: bool=True) -> typing.Tuple[list, ...]:
    '''
    Load the dataset given the directory path.

    inputs:
        dataset_dir  -> (str) the path to the dataset directory containing all files (requires backslash at the end!)
        clean_labels -> (bool, optional) if set, will modify the labels into onehot
    outputs:
        ...          -> (tuple) a collection of all dataset splits
    '''
    if ("IMPLI" in dataset_dir):
        # Load in all splits of the IMPLI benchmark
        magpie_e = pd.read_csv(dataset_dir + "fig_context_magpie_e.tsv", sep="\t").values.tolist()
        magpie_ne = pd.read_csv(dataset_dir + "lit_context_magpie_ne.tsv", sep="\t").values.tolist()
        magpie_adversarial_ne = pd.read_csv(dataset_dir + "adversarial_definition_ne_magpie.tsv", sep="\t").values.tolist()
        
        pie_e = pd.read_csv(dataset_dir + "fig_context_pie_e.tsv", sep="\t").values.tolist()
        pie_ne = pd.read_csv(dataset_dir + "lit_context_pie_ne.tsv", sep="\t").values.tolist()
        pie_adversarial_ne = pd.read_csv(dataset_dir + "adversarial_definition_ne_pie.tsv", sep="\t").values.tolist()
            
        semeval_e = pd.read_csv(dataset_dir + "fig_context_semeval_e.tsv", sep="\t").values.tolist()
        semeval_ne = pd.read_csv(dataset_dir + "lit_context_semeval_ne.tsv", sep="\t").values.tolist()
        semeval_adversarial_ne = pd.read_csv(dataset_dir + "adversarial_definition_ne_semeval.tsv", sep="\t").values.tolist()
        
        manual_e = pd.read_csv(dataset_dir + "manual_e.tsv", sep="\t", header=None).values.tolist()
        manual_ne = pd.read_csv(dataset_dir + "manual_ne.tsv", sep="\t", header=None).values.tolist()
        manual_antonyms_ne = pd.read_csv(dataset_dir + "manual_antonyms_ne.tsv", sep="\t", header=None).values.tolist()

        if (clean_labels == True):
            magpie_e, magpie_ne, magpie_adversarial_ne = label2onehot(entailment=magpie_e, 
                                                                      nonentailment=magpie_ne, 
                                                                      adversarial=magpie_adversarial_ne)
            pie_e, pie_ne, pie_adversarial_ne = label2onehot(entailment=pie_e, 
                                                             nonentailment=pie_ne, 
                                                             adversarial=pie_adversarial_ne)
            semeval_e, semeval_ne, semeval_adversarial_ne = label2onehot(entailment=semeval_e, 
                                                                         nonentailment=semeval_ne, 
                                                                         adversarial=semeval_adversarial_ne)
            manual_e, manual_ne, manual_antonyms_ne = label2onehot(entailment=manual_e, 
                                                                   nonentailment=manual_ne, 
                                                                   adversarial=manual_antonyms_ne,
                                                                   construct=True)

        train_ent = magpie_e + pie_e + semeval_e
        train_nent = magpie_ne + pie_ne + semeval_ne
        train_ant = magpie_adversarial_ne + pie_adversarial_ne + semeval_adversarial_ne

        # Filter samples by length
        train_ent = length_filtering(dataset=train_ent, min_length=6, verbose=False, invalid_labels=[])
        train_nent = length_filtering(dataset=train_nent, min_length=6, verbose=False, invalid_labels=[])
        train_ant = length_filtering(dataset=train_ant, min_length=6, verbose=False, invalid_labels=[])
        
        return train_ent, train_nent, train_ant, manual_e, manual_ne, manual_antonyms_ne
    elif ("FigurativeNarrativeBenchmark" in dataset_dir):
        # Load in all splits of the FigurativeNarrativeBenchmark dataset
        fnb_train = pd.read_json(path_or_buf=dataset_dir + "train.jsonl", lines=True).values.tolist()
        fnb_test = pd.read_json(path_or_buf=dataset_dir + "test.jsonl", lines=True).values.tolist()
    
        # Key format: background, idiom definition, idiom, option1, option2, ground truth selection
        return fnb_train, fnb_test
    else:
        print(f"Dataset was not found with input directory: {dataset_dir}")
        return None

def clean_text(text: str) -> str:
    '''
    Given a text string, fix any general formatting issues.

    input:
        text -> (str) the text sample to fix
    output:
        text -> (str) the fixed text sample
    '''
    if (text == ""): return text

    # General formatting fixes
    text = text.replace("  </s>", ". </s>") # format EOS tokens appropriately
    text = text.replace(" i ", " I ") # capitalize single I
    text = text.replace("_", " ") # replace underscores
    text = text.replace("‘ ", "") # remove special character

    text = text.replace("..", ".") # remove double periods
    text = text.replace(" . ", ".") # fix period spacing
    text = text.replace(" .", ".") # remove spaces before periods
    text = text.replace(" ?", "?") # remove spaces before question marks

    text = text.replace("  ", ", ") # replace double spaces with commas
    text = text.replace(" , ", ", ") # fix comma spacing
    text = text.replace(" - ", "-") # remove spaces around a dash
    text = text.replace("-", " ") # then remove dashes
    text = text.replace(",I", ", I") # fix comma spacing
    text = text.replace(" ; ", "; ") # fix semicolon spacing

    text = text.replace(" 'll", "'ll") # fix apostrophes
    text = text.replace("I 'd", "I'd") # fix apostrophes
    text = text.replace("I' ll", "I'll") # fix apostrophes
    text = text.replace(" 's", "'s") # fix apostrophes
    
    text = text.replace("ca n't", "can't") # fix conjunction
    text = text.replace("they 'd", "they'd") # fix conjunction
    text = text.replace("( ", "(") # remove spaces after open parentheses
    text = text.replace(" )", ")") # remove spaces before closed parentheses
    
    # Capitalize the first letter
    if (len(text) > 1): text = text[0].upper() + text[1:] # Capitalise sentence
    else: text = text.upper()

    # Remove any extra spaces
    text = text.strip()
    
    # Add period at the end of the text
    if (text == "" or (text.strip()[-1] not in [".", "!", "?"])): text += "."

    return text

# Context processing methods
def remove_context(text: typing.Union[str, list], dataset_name: str, amount2keep: float=None, total_removal: bool=False, no_removal: bool=False) -> typing.Union[str, typing.Tuple[str, str]]:
    '''
    Given an input text with the corresponding dataset name, remove the context according to the desired parameters.

    inputs:
        text          -> (str | list) the original text sample (or list of text samples) from which we hope to remove the context from
        dataset_name  -> (str) specifies how to proceed with context removal, as different datasets will operate slightly differently
        amount2keep   -> (float, optional) if set, will perform a percentage removal (really only useful for longer texts like FNB)
        total_removal -> (bool, optional) if set, will completely remove all text that does not correspond to the idiom
        no_removal    -> (bool, optional) if set, will override ALL OTHER parameters and just return the input string
    outputs:
        text          -> (str) the context removed text sample
    '''
    if (no_removal == True): return text
    
    if (dataset_name == "FNB"): # signals FigurativeNarrativeBenchmark dataset
        # Return just the last sentence, where the idiom is located for the FigurativeNarrativeBenchmark dataset
        if (total_removal == True):
            if ("<b>" not in premise): return premise.split(". ")[-1] 
            text = text.split("<b>")[1].split("</b>")[0]
            return text[0]

        # Perform a percentage removal of words starting from the beginning (i.e. furthest from the idiom)
        if (amount2keep != None and amount2keep > 0.0 and amount2keep < 1.0):
            all_words = text.split(" ")
            keep = int(amount2keep * len(all_words))
            text = " ".join(all_words[-keep:])
            return text
        
        # Default is to remove all sentences except for the one containing the idiom (typically the last one)
        text, idx = text.split(". "), -1        
        
        for i in range(len(text)):
            if ("<b>" in text[i]): 
                idx = i
                break
            pass

        text = text[idx]        
        return text
    elif (dataset_name == "IMPLI"): # signals IMPLI dataset
        s1, s2 = text[0].split(" "), text[1].split(" ") # extract the premise and hypothesis
        
        # Cutoff the left portion of the text
        l, min_length = -1, min(len(s1), len(s2))
        for i in range(min_length):
            # Given the format of IMPLI, if the characters differ, then it signals the start of the idiom
            if (s1[i].lower() != s2[i].lower()):
                l = i
                break
            pass
        s1, s2 = s1[l:], s2[l:]

        # Cutoff the right portion of the text
        r, min_length = -1, min(len(s1), len(s2))
        for i in range(1, min_length):
            if (s1[-i].lower() != s2[-i].lower()):
                r = -i
                break
            pass
        if (r != -1): s1, s2 = s1[:r+1], s2[:r+1]
    
        return " ".join(s1), " ".join(s2) 
    else:
        print(f"Error -> could not find dataset name: {dataset_name}.")
        return text

def context_shuffling(text: str, dataset_name: str, impli_df: pd.DataFrame=IMPLI_DF, impli_idx: int=None) -> str:
    '''
    Given an input text with the corresponding dataset name, shuffle the context.

    inputs:
        text          -> (str) the original text sample from which we hope to shuffle the context from
        dataset_name  -> (str) specifies how to proceed with context shuffling, as different datasets will operate slightly differently
        impli_df      -> (pd.DataFrame, optional) if set with the IMPLI dataset, will use it for shuffling IMPLI text
        impli_idx     -> (int, optional) if set with the IMPLI dataset, will use it for shuffling IMPLI text
    outputs:
        shuffled_text -> (str) the context removed text sample
    '''
    if (dataset_name == "FNB"): # signals FigurativeNarrativeBenchmark dataset
        text = text.split(" ")
        
        # Get the range of indices corresponding to the idiom itself
        start, end = 0, -1
        
        for i in range(len(text)):
            if ((text[i] in ["<b>", "b>", "<b"]) and (start == 0)): start = i
            if ((text[i] in ["</b>", "/b>", "</b"]) and (end == -1)): end = i
            pass
        
        if (end != len(text) - 1): end += 1

        # Shuffle the non-idiom portions of the text
        start_portion, end_portion = text[:start], text[end:]
        
        # Shuffle both portions
        if (len(start_portion) > 0): random.shuffle(start_portion)
        if (len(end_portion) > 0): random.shuffle(end_portion)
        
        # Return the final, context shuffled, output
        shuffled_text = start_portion + text[start:end] + end_portion
        shuffled_text = (" ".join(shuffled_text)).strip()
        return shuffled_text    
    elif (dataset_name == "IMPLI"): # signals IMPLI dataset
        if (impli_df is None or impli_idx == None):
            print(f"Error -> either the IMPLI dataframe or idx was not set.")
            return text

        idiom, text = impli_df.iloc[impli_idx]["idiom"].split(" "), text.split(" ")

        # Search for a fuzzy string match of the idiom
        completed, j = False, 0 # j is the idiom traversal index
        start, end = 0, -1

        # Nested for-loops to test for all possible start locations
        for i in range(len(text)):
            score = fuzzywuzzy.fuzz.ratio(text[i].lower(), idiom[j].lower())

            # If no match was found, then continue, otherwise mark the starting index
            if (score < 75): continue
            start = i

            # If the idiom is a single word (unlikely), return just that
            if (len(idiom) == 1): 
                end, completed = i+1, True
                break

            for k in range(i+1, len(text)):
                score = fuzzywuzzy.fuzz.ratio(text[k].lower(), idiom[k-i].lower())
                if (score < 75): break

                if (k-i == len(idiom) - 1):
                    end, completed = k+1, True
                    break
                pass
            if (completed == True): break
            pass

        # If nothing was found, just return the original text
        if (start == 0 and end == -1): return " ".join(text)

        # Split the text into different parts and shuffle accordingly
        start_portion, end_portion = text[:start], text[end:]
        random.shuffle(start_portion)
        random.shuffle(end_portion)
        shuffled_text = " ".join(start_portion+text[start:end]+end_portion).strip()
        return shuffled_text
    else:
        print(f"Error -> could not find dataset name: {dataset_name}.")
        return text

def random_context_removal(text: str, dataset_name: str, amount2keep: float=0.9) -> str:
    '''
    Given an input text with the corresponding dataset name, randomly remove a portion of the context.

    inputs:
        text         -> (str) the original text sample from which we hope to randomly remove context from
        dataset_name -> (str) specifies how to proceed with random context removal, as different datasets will operate slightly differently
        amount2keep  -> (float, optional) if set, will keep a percentage of the text (really only useful for longer texts like FNB)
    outputs:
        text         -> (str) the random context removed text sample
    '''
    if (dataset_name == "FNB"): # signals FigurativeNarrativeBenchmark dataset
        # Compute the number of words to keep and remove
        text = text.split(" ")
        num_words = len(text)
        num_keep = int(num_words * amount2keep)
        num_remove = num_words - num_keep
        
        # Get the range of indices corresponding to the idiom itself
        start, end = 0, -1
        
        for i in range(len(text)):
            if ((text[i] in ["<b>", "b>", "<b"]) and (start == 0)): start = i
            if ((text[i] in ["</b>", "/b>", "</b"]) and (end == -1)): end = i
            pass

        if (end != len(text) - 1): end += 1
            
        # Get the indices of tokens to remove
        idx = []
        
        for i in range(num_remove):
            removal_idx = np.random.randint(num_words)
            while (removal_idx >= start and removal_idx < end): removal_idx = np.random.randint(num_words)
            idx.append(removal_idx)
            pass
        
        text = [j for i, j in enumerate(text) if i not in idx]
        text = " ".join(text).strip()
        return text
    else:
        print(f"Error -> could not find dataset name: {dataset_name}.")
        return text

def find_idiom(text: str, dataset_name: str) -> str:
    '''
    Given the original text sample, extract the actual idiom from the text.

    inputs:
        text         -> (str) the original text sample from which we hope to extract the idiom from
        dataset_name -> (str) specifies how to proceed with idiom extraction, as different datasets will operate slightly differently
    outputs:
        idiom        -> (str) the extracted idiom
    '''
    if (dataset_name == "FNB"): # signals FigurativeNarrativeBenchmark dataset
        if ("<b>" not in text): return text.split(". ")[-1] # The last sentence
        idiom = premise.split("<b>")[1].split("</b>")[0]
        return idiom
    else:
        print(f"Error -> could not find dataset name: {dataset_name}.")
        return text

def generate_gibberish(threshold: int=15, valid_chars: str="abcdefghijklmnopqrstuvwxyz ") -> str:
    '''
    Given the function parameters, generate a random gibberish string.

    inputs:
        threshold   -> (int, optional) the maximum cutoff threshold for the gibberish string
        valid_chars -> (str, optional) a string containing all possible characters for our gibberish
    outputs:
        gibberish   -> (str) the gibberish text
    '''
    length = np.random.randint(threshold)

    # Generate gibberish by sampling from the valid characters inputted
    gibberish = ""
    for i in range(length): gibberish += random.choice(valid_chars)
    return gibberish

def idiom2gibberish(dataset: list, dataset_name: str, threshold: int=15, valid_chars: str="abcdefghijklmnopqrstuvwxyz ") -> list:
    '''
    Given the original text sample, replace the idiom with gibberish according to the input parameters.

    input:
        dataset      -> (list) a list of all dataset samples
        dataset_name -> (str) specifies how to proceed with gibberish generation, as different datasets will operate slightly differently
        threshold    -> (int, optional) the maximum cutoff threshold for the gibberish string
        valid_chars  -> (str, optional) a string containing all possible characters for our gibberish
    outputs:
        dataset      -> (list) the dataset with the idioms replaced by some gibberish strings
    '''
    if (dataset_name == "FNB"): # signals FigurativeNarrativeBenchmark dataset
        for i in range(len(dataset)):
            idiom = find_idiom(text=dataset[i][0], dataset_name=dataset_name)
            gibberish = generate_gibberish(threshold=threshold, valid_chars=valid_chars)
            dataset[i][0] = dataset[i][0].replace(idiom, gibberish)
            pass

        return dataset
    elif (dataset_name == "IMPLI"): # signals IMPLI dataset
        for i in range(len(dataset)):
            gibberish = generate_gibberish(threshold=threshold, valid_chars=valid_chars)

            # Replace the idiom with gibberish
            idiom = remove_context(text=[dataset[i][0], dataset[i][1]], dataset_name="IMPLI")[0]
            dataset[i][0] = dataset[i][0].replace(idiom, gibberish)
            pass
        
        return dataset
    else:
        print(f"Error -> could not find dataset name: {dataset_name}.")
        return dataset

def construct_fnb(train: list, test: list, amount2keep: float=None, total_removal: bool=False, no_removal: bool=False, shuffle: bool=False, seed: int=42) -> datasets.Dataset:
    '''
    Given the train and test splits, construct our FigurativeNarrativeBenchmark dataset.

    inputs:
        train         -> (list) a list of the training samples for the dataset
        test          -> (list) a list of the test samples for our dataset
        amount2keep   -> (float, optional) if set, will keep a percentage of the text
        total_removal -> (bool, optional) if set, will completely remove all text that does not correspond to the idiom
        no_removal    -> (bool, optional) if set, will not remove any context in the text
        shuffle       -> (bool, optional) if set, will shuffle the context of the text (assuming no context removal)
        seed          -> (int, optional) the random seed from which to shuffle any training samples
    outputs:
        final_train   -> (datasets.Dataset) a dataset of all train split examples
        final_test    -> (datasets.Dataset) a dataset of all test split examples
        test_ent      -> (datasets.Dataset) a dataset of the entailment test split examples
        test_nent     -> (datasets.Dataset) a dataset of the nonentailment test split examples
    '''
    # Construct the train split
    final_train = {"premises" : [], "hypotheses" : [], "labels" : []}
    
    for sample in train:
        label = sample[3]

        # Recall that each original input will have 2 samples, a correct and an incorrect one
        final_train["premises"].append(clean_text(sample[0]))
        final_train["hypotheses"].append(clean_text(sample[1]))
        final_train["premises"].append(clean_text(sample[0]))
        final_train["hypotheses"].append(clean_text(sample[2]))

        # Append the labels according to the ground truth option
        if (label == "option1"):
            final_train["labels"].append([1.0, 0.0])
            final_train["labels"].append([0.0, 1.0])
        else:
            final_train["labels"].append([0.0, 1.0])
            final_train["labels"].append([1.0, 0.0])
        pass

    # Construct the test splits
    final_test, test_ent, test_nent = {"premises" : [], "hypotheses" : [], "labels" : []}, {"premises" : [], "hypotheses" : [], "labels" : []}, {"premises" : [], "hypotheses" : [], "labels" : []}

    for sample in test:
        label = sample[3]

        # Fix the premise according to our parameters
        premise = remove_context(text=sample[0], 
                                 dataset_name="FNB", 
                                 amount2keep=amount2keep, 
                                 total_removal=total_removal,
                                 no_removal=no_removal)

        # Shuffle the context if none was removed
        if (shuffle == True and no_removal == True): premise = context_shuffling(text=premise, dataset_name="FNB")

        # Add our examples to the datasets
        final_test["premises"].append(clean_text(premise))
        final_test["hypotheses"].append(clean_text(sample[1]))
        final_test["premises"].append(clean_text(premise))
        final_test["hypotheses"].append(clean_text(sample[2]))
        test_ent["premises"].append(clean_text(premise))
        test_ent["labels"].append([1.0, 0.0])
        test_nent["premises"].append(clean_text(premise))
        test_nent["labels"].append([0.0, 1.0])

        # Append the labels and test specific splits according to the ground truth option
        if (label == "option1"):
            final_test["labels"].append([1.0, 0.0])
            final_test["labels"].append([0.0, 1.0])
            test_ent["hypotheses"].append(clean_text(sample[1]))
            test_nent["hypotheses"].append(clean_text(sample[2]))
        else:
            final_test["labels"].append([0.0, 1.0])
            final_test["labels"].append([1.0, 0.0])
            test_ent["hypotheses"].append(clean_text(sample[2]))
            test_nent["hypotheses"].append(clean_text(sample[1]))
        pass
    
    final_train, final_test, test_ent, test_nent = datasets.Dataset.from_dict(final_train).shuffle(seed=seed), datasets.Dataset.from_dict(final_test), datasets.Dataset.from_dict(test_ent), datasets.Dataset.from_dict(test_nent)    
    return final_train, final_test, test_ent, test_nent

def construct_impli(train_ent: list, train_nent: list, train_ant: list, test_ent: list, test_nent: list, test_ant: list, impli_df: pd.DataFrame=IMPLI_DF, balance: bool=True, no_context: bool=True, shuffling: bool=True, seed: int=42) -> typing.Tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset, datasets.Dataset, datasets.Dataset]:
    '''
    Given the entailment, nonentailment, and adversarial nonentailment samples for both the train and test splits, construct the IMPLI dataset.

    inputs:
        train_ent     -> (list) a list of entailment samples from the train split
        train_nent    -> (list) a list of nonentailment samples from the train split
        train_ant     -> (list) a list of antonym nonentailment samples from the train split
        test_ent      -> (list) a list of entailment samples from the test split
        test_nent     -> (list) a list of nonentailment samples from the test split
        test_ant      -> (list) a list of antonym nonentailment samples from the test split
        impli_df      -> (pd.DataFrame) a detailed dataframe of the IMPLI samples
        balance       -> (bool, optional) if set, will balance the classes in the training split
        no_context    -> (bool, optional) if set, will strip the context for each sample
        shuffling     -> (bool, optional) if set, will shuffle the context for each sample, provided no context was removed
        seed          -> (int, optional) the random seed from which to shuffle any training samples
    outputs:
        final_train   -> (datasets.Dataset) a dataset of all train split examples
        final_test    -> (datasets.Dataset) a dataset of all test split examples
        final_ent     -> (datasets.Dataset) a dataset of the entailment test split examples
        final_nent    -> (datasets.Dataset) a dataset of the nonentailment test split examples
        final_ant     -> (datasets.Dataset) a dataset of the antonym nonentailment test split examples
    '''
    if (balance == True):
        num_nent = len(train_nent) + len(train_ant)
        idx = random.sample(range(len(train_ent)), num_nent)
        train_ent = [train_ent[index] for index in idx]

    # Construct the entire raw training dataset
    aggregate_train = train_ent + train_nent + train_ant
    final_train = datasets.Dataset.from_dict({"premises" : [clean_text(s[0]) for s in aggregate_train],
                                              "hypotheses" : [clean_text(s[1]) for s in aggregate_train],
                                              "labels" : [s[2] for s in aggregate_train]}).shuffle(seed=seed)

    # Construct the test dataset
    aggregate_test, final_test = test_ent + test_nent + test_ant, {"premises" : [], "hypotheses" : [], "labels" : []}
    final_ent, final_nent, final_ant = {"premises" : [], "hypotheses" : [], "labels" : []}, {"premises" : [], "hypotheses" : [], "labels" : []}, {"premises" : [], "hypotheses" : [], "labels" : []}
    
    for idx, sample in enumerate(aggregate_test):
        if (no_context == True): premise, hypothesis = remove_context(text=[clean_text(sample[0]), clean_text(sample[1])], dataset_name="IMPLI")
        elif (shuffling == True): premise, hypothesis = context_shuffling(text=clean_text(sample[0]), dataset_name="IMPLI", impli_df=impli_df, impli_idx=idx), clean_text(sample[1])
        else: premise, hypothesis = clean_text(sample[0]), clean_text(sample[1])

        final_test["premises"].append(premise)
        final_test["hypotheses"].append(hypothesis)
        final_test["labels"].append(sample[2])
        pass

    # Construct the individual test splits
    for idx, sample in enumerate(test_ent):
        if (no_context == True): premise, hypothesis = remove_context(text=[clean_text(sample[0]), clean_text(sample[1])], dataset_name="IMPLI")
        elif (shuffling == True): premise, hypothesis = context_shuffling(text=clean_text(sample[0]), dataset_name="IMPLI", impli_df=impli_df, impli_idx=idx), clean_text(sample[1])
        else: premise, hypothesis = clean_text(sample[0]), clean_text(sample[1])

        final_ent["premises"].append(premise)
        final_ent["hypotheses"].append(hypothesis)
        final_ent["labels"].append(sample[2])
        pass
    for idx, sample in enumerate(test_nent):
        if (no_context == True): premise, hypothesis = remove_context(text=[clean_text(sample[0]), clean_text(sample[1])], dataset_name="IMPLI")
        elif (shuffling == True): premise, hypothesis = context_shuffling(text=clean_text(sample[0]), dataset_name="IMPLI", impli_df=impli_df, impli_idx=idx), clean_text(sample[1])
        else: premise, hypothesis = clean_text(sample[0]), clean_text(sample[1])

        final_nent["premises"].append(premise)
        final_nent["hypotheses"].append(hypothesis)
        final_nent["labels"].append(sample[2])
        pass
    for idx, sample in enumerate(test_ant):
        if (no_context == True): premise, hypothesis = remove_context(text=[clean_text(sample[0]), clean_text(sample[1])], dataset_name="IMPLI")
        elif (shuffling == True): premise, hypothesis = context_shuffling(text=clean_text(sample[0]), dataset_name="IMPLI", impli_df=impli_df, impli_idx=idx), clean_text(sample[1])
        else: premise, hypothesis = clean_text(sample[0]), clean_text(sample[1])

        final_ant["premises"].append(premise)
        final_ant["hypotheses"].append(hypothesis)
        final_ant["labels"].append(sample[2])
        pass

    # Create the test datasets
    final_test, final_ent, final_nent, final_ant = datasets.Dataset.from_dict(final_test), datasets.Dataset.from_dict(final_ent), datasets.Dataset.from_dict(final_nent), datasets.Dataset.from_dict(final_ant)
    
    # Return all datasets
    return final_train, final_test, final_ent, final_nent, final_ant

# Training methods
def tokenize_text(examples: datasets.Dataset, tokenizer: transformers.AutoTokenizer, truncation: bool=True, padding: str="max_length", max_length: int=256) -> transformers.tokenization_utils_base.BatchEncoding:
    '''
    Tokenize text samples in our dataset

    inputs:
        examples       -> (datasets.Dataset) the text dataset
        tokenizer      -> (transformers.AutoTokenizer) a tokenizer to preprocess our text samples
        truncation     -> (bool, optional) if set, will truncate inputs that exceed the maximum length
        padding        -> (str, optional) specifies how to pad the input text
        max_length     -> (int, optional) a parameter that specifies the max length for the padding
    outputs:
        tokenized_text -> (transformers.tokenization_utils_base.BatchEncoding) tokenized version of the input text
    '''
    tokenized_text = tokenizer(examples["premises"], examples["hypotheses"], truncation=True, max_length=max_length, padding="max_length")
    return tokenized_text

def compute_metrics(eval_preds: transformers.trainer_utils.EvalPrediction) -> dict:
    '''
    Evaluate the quality of model predictions compared to the ground truth labels

    inputs:
        eval_preds -> (transformers.trainer_utils.EvalPrediction) the output trainer logits and labels for a particular batch
    outputs:
        results    -> (dict) a dictionary of results that we computed
    '''
    predictions, labels = eval_preds
    if isinstance(predictions, tuple): predictions = predictions[0]
    predictions = np.argmax(predictions, axis=-1)
    labels = [abs(label[0] - 1) for label in labels]
    results = {"accuracy" : ACCURACY.compute(predictions=predictions, references=list(labels))}
    return results

def trainNtest(model: transformers.AutoModelForSequenceClassification, tokenizer: transformers.AutoTokenizer, training_args: dict, train_dataset: datasets.Dataset, test_datasets: list, skip_train: bool=False) -> transformers.AutoModelForSequenceClassification:
    '''
    Train the given model on the train dataset and evaluates on the evaluation dataset.

    inputs:
        model         -> (transformers.AutoModelForSequenceClassification) the model we want to train
        tokenizer     -> (transformers.AutoTokenizer) the tokenizer that corresponds to the model
        training_args -> (dict) a dictionary containing all the possible training arguments
        train_dataset -> (datasets.Dataset) the dataset to train our model on
        test_datasets -> (list) a list of datasets we want to evaluate on
        skip_train    -> (bool, optional) if set, will skip training completely and go onto evaluation
    outputs:
        model         -> (transformers.AutoModelForSequenceClassification) the trained version of the model
    '''
    # Use default training arguments when not present
    keys = ["output_dir", "evaluation_strategy", "epochs", "lr", "weight_decay", "warmup_steps", "batch_size", "fp16"]
    vals = ["/tmp/", "no", 5, 2e-5, 0.01, 0, 32, True]
    for idx, key in enumerate(keys):
        try: training_args[key]
        except: training_args[key] = vals[idx]
        pass

    # Setup the training arguments and trainer
    args = transformers.Seq2SeqTrainingArguments(output_dir=training_args["output_dir"],
                                                 evaluation_strategy=training_args["evaluation_strategy"],
                                                 num_train_epochs=training_args["epochs"],
                                                 learning_rate=training_args["lr"],
                                                 weight_decay=training_args["weight_decay"],
                                                 warmup_steps=training_args["warmup_steps"],
                                                 per_device_train_batch_size=training_args["batch_size"],
                                                 per_device_eval_batch_size=training_args["batch_size"],
                                                 fp16=training_args["fp16"],
                                                 push_to_hub=False)
    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer, max_length=256)
    trainer = transformers.Seq2SeqTrainer(model=model, 
                                          args=args,
                                          train_dataset=train_dataset,
                                          eval_dataset=test_datasets[0],
                                          tokenizer=tokenizer,
                                          compute_metrics=compute_metrics)

    # Train the model
    if (skip_train == False): trainer.train()

    # Evaluate all the datasets
    for i in range(len(test_datasets)):
        print(trainer.evaluate(test_datasets[i]))
        print("="*100)
        pass
        
    return model