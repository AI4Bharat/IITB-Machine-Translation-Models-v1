from indicnlp.tokenize import sentence_tokenize, indic_detokenize, indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

import codecs
from subword_nmt.apply_bpe import BPE

import ctranslate2


## Normalize
factory=IndicNormalizerFactory()
normalizer_hi=factory.get_normalizer("hi")
normalizer_mr=factory.get_normalizer("mr")

bpe_hi = BPE(codecs.open("mt-models/hi-mr/hi-mr/v1.0/bpe-codes/codes.hi", encoding='utf-8'))
bpe_mr = BPE(codecs.open("mt-models/hi-mr/mr-hi/v1.0/bpe-codes/codes.mr", encoding='utf-8'))

translator_himr = ctranslate2.Translator("mt-models/hi-mr/hi-mr/v1.0/model_ct2/", inter_threads=4, intra_threads=1)
translator_mrhi = ctranslate2.Translator("mt-models/hi-mr/mr-hi/v1.0/model_ct2/", inter_threads=4, intra_threads=1)


def translate_hi_to_mr(sentence):

    inp_text = sentence.strip("\n")

    # Normalize
    inp_text = normalizer_hi.normalize(inp_text)

    # Tokenize
    inp_text = ' '.join(indic_tokenize.trivial_tokenize(inp_text))

    # Apply BPE
    inp_text = bpe_hi.process_line(inp_text).split(" ")

    # Translate
    out_text = translator_himr.translate_batch([inp_text], beam_size=5, max_batch_size=16)

    # Remove BPE
    out_text = (' '.join(out_text[0].hypotheses[0]) + " ").replace("@@ ", "")

    # Post Processing
    out_text = out_text.replace('"', '').replace("u200d", "").strip()

    # Detokenize
    out_text = indic_detokenize.trivial_detokenize(out_text)

    return out_text

def translate_mr_to_hi(sentence):

    inp_text = sentence.strip("\n")

    # Normalize
    inp_text = normalizer_mr.normalize(inp_text)

    # Tokenize
    inp_text = ' '.join(indic_tokenize.trivial_tokenize(inp_text))

    # Apply BPE
    inp_text = bpe_mr.process_line(inp_text).split(" ")

    # Translate
    out_text = translator_mrhi.translate_batch([inp_text], beam_size=5, max_batch_size=16)

    # Remove BPE
    out_text = (' '.join(out_text[0].hypotheses[0]) + " ").replace("@@ ", "")

    # Post Processing
    out_text = out_text.replace('"', '').replace("u200d", "").strip()

    out_text = indic_detokenize.trivial_detokenize(out_text)

    return out_text


# print(translate_hi_to_mr("पेड़ बहुत ऊंचा है।"))
# print(translate_mr_to_hi("वृक्ष खूप उंच आहे."))