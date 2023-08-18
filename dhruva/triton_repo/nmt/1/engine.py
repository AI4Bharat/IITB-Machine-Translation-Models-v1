import os
from sacremoses import MosesTokenizer
from sacremoses import MosesDetokenizer
from subword_nmt.apply_bpe import BPE
import codecs

from indicnlp.tokenize import indic_tokenize
from indicnlp.tokenize import indic_detokenize
from indicnlp.normalize import indic_normalize

from mosestokenizer import MosesSentenceSplitter
from indicnlp.tokenize.sentence_tokenize import sentence_split, DELIM_PAT_NO_DANDA

from ctranslate2 import Translator


INDIC_LANGUAGES = {"as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"}


def split_sentences(paragraph, language):
    if language == "en":
        with MosesSentenceSplitter(language) as splitter:
            return splitter([paragraph])
    elif language in INDIC_LANGUAGES:
        return sentence_split(paragraph, lang=language, delim_pat=DELIM_PAT_NO_DANDA)

def truncate_long_sentences(sents):

    MAX_SEQ_LEN = 200
    new_sents = []

    for sent in sents:
        words = sent.split()
        num_words = len(words)
        if num_words > MAX_SEQ_LEN:
            print_str = " ".join(words[:5]) + " .... " + " ".join(words[-5:])
            sent = " ".join(words[:MAX_SEQ_LEN])
            print(
                f"WARNING: Sentence {print_str} truncated to 200 tokens as it exceeds maximum length limit"
            )

        new_sents.append(sent)
    return new_sents


class Model:
    def __init__(self, ckpt_dir, src_lang, tgt_lang, device = "cuda"):
        self.ckpt_dir = ckpt_dir
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.en_tok = MosesTokenizer(lang="en")
        self.en_detok = MosesDetokenizer(lang="en")
        
        print("Initializing bpe")
        self.bpe = BPE(
            codecs.open(f"{ckpt_dir}/bpe-codes/codes.{src_lang}", encoding='utf-8')
        )

        print("Initializing model for translation")
        self.translator = Translator(os.path.join(self.ckpt_dir, "model_ct2"), device=device)

    def batch_translate(self, batch):

        assert isinstance(batch, list)
        preprocessed_sents = self.preprocess(batch, lang=self.src_lang)
        bpe_sents = self.apply_bpe(preprocessed_sents)
        bpe_sents = truncate_long_sentences(bpe_sents)

        bpe_tokens = [x.strip().split(" ") for x in bpe_sents]
        translated_tokens = self.translator.translate_batch(bpe_tokens, beam_size=5, max_batch_size=16)
        translated_tokens = [x.hypotheses[0] for x in translated_tokens]
        bpe_translations = [" ".join(x) + " " for x in translated_tokens]
        translations = [x.replace("@@ ", "") for x in bpe_translations]

        postprocessed_sents = self.postprocess(translations, self.tgt_lang)
        return postprocessed_sents

    def translate_paragraph(self, paragraph):

        assert isinstance(paragraph, str)
        sents = split_sentences(paragraph, self.src_lang)

        postprocessed_sents = self.batch_translate(sents)

        translated_paragraph = " ".join(postprocessed_sents)

        return translated_paragraph

    def preprocess_sent(self, sent, normalizer, lang):
        if lang == "en":
            return " ".join(
                self.en_tok.tokenize(
                    sent.strip().lower()
                )
            )
        else:
            return " ".join(
                indic_tokenize.trivial_tokenize(
                    normalizer.normalize(sent.strip()), lang
                )
            ).replace(" ् ", "्")

    def preprocess(self, sents, lang):

        if lang == "en":
            processed_sents = [
                self.preprocess_sent(line, None, lang) for line in sents
            ]
        else:
            normalizer = indic_normalize.IndicNormalizerFactory().get_normalizer(lang)

            processed_sents = [
                self.preprocess_sent(line, normalizer, lang) for line in sents
            ]

        return processed_sents

    def postprocess(self, sents, lang):
        postprocessed_sents = []

        if lang == "en":
            for sent in sents:
                postprocessed_sents.append(
                    self.en_detok.detokenize(sent.split(" ")).capitalize()
                )
        else:
            for sent in sents:
                outstr = indic_detokenize.trivial_detokenize(
                    sent.replace('"', '').replace("u200d", "").strip()
                )
                postprocessed_sents.append(outstr)
        return postprocessed_sents

    def apply_bpe(self, sents):
        return [self.bpe.process_line(sent) for sent in sents]
