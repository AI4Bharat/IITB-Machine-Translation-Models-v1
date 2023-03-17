from mosestokenizer import MosesSentenceSplitter, MosesTokenizer, MosesDetokenizer

from indicnlp.tokenize import sentence_tokenize, indic_tokenize, indic_detokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

import codecs
from subword_nmt.apply_bpe import BPE

import ctranslate2

# En-Mr
## Split
splitsents_en = MosesSentenceSplitter('en')
## Tokenize
tokenize_en = MosesTokenizer('en')
detokenize_en = MosesDetokenizer('en')
## BPE
codes_en = codecs.open("mt-models/en-mr/en-mr/v1.0/bpe-codes/codes.en", encoding='utf-8')
bpe_en = BPE(codes_en)
## Translate
translator_enmr = ctranslate2.Translator("mt-models/en-mr/en-mr/v1.0/model_deploy/", inter_threads=4, intra_threads=1)

# Mr-En
# Normalize
factory=IndicNormalizerFactory()
normalizer_mr=factory.get_normalizer("mr")
## BPE
codes_mr = codecs.open("mt-models/en-mr/mr-en/v1.0/bpe-codes/codes.mr", encoding='utf-8')
bpe_mr = BPE(codes_mr)
## Translate
translator_mren = ctranslate2.Translator("mt-models/en-mr/mr-en/v1.0/model_deploy/", inter_threads=4, intra_threads=1)


def translate_en_to_mr(sentence):

    inp_text = sentence.strip("\n")

    # Lowercase
    inp_text = inp_text.lower()
    
    # Tokenize
    inp_text = ' '.join(tokenize_en(inp_text))

    # Apply BPE
    inp_text = bpe_en.process_line(inp_text).split(" ")

    # Translate
    out_text = translator_enmr.translate_batch([inp_text], beam_size=5, max_batch_size=16)

    # Remove BPE
    out_text = (' '.join(out_text[0].hypotheses[0]) + " ").replace("@@ ", "")

    # Post Processing
    out_text = out_text.replace('"', '').replace("u200d", "").strip()

    # Detokenize
    out_text = indic_detokenize.trivial_detokenize(out_text)

    return out_text    

def translate_mr_to_en(sentence):

    inp_text = sentence.strip("\n")

    # Normalize
    inp_text = normalizer_mr.normalize(inp_text)

    # Tokenize
    inp_text = ' '.join(indic_tokenize.trivial_tokenize(inp_text))

    # Apply BPE
    inp_text = bpe_mr.process_line(inp_text).split(" ")

    # Translate
    out_text = translator_mren.translate_batch([inp_text], beam_size=5, max_batch_size=16)

    # Remove BPE
    out_text = (' '.join(out_text[0].hypotheses[0]) + " ").replace("@@ ", "").replace("u200d",'').replace('"', '').strip()

    # Detokenize
    out_text = detokenize_en(out_text.split(" "))
    
    # Capitalize
    out_text = out_text.capitalize()

    return out_text

# print(translate_en_to_mr("The tree is very tall."))
# print(translate_mr_to_en("झाड खूप उंच आहे."))

