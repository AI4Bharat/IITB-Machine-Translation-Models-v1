import tritonclient.http as http_client
from tritonclient.utils import *

ENDPOINT_URL = 'localhost:8000'
triton_http_client = http_client.InferenceServerClient(
    url=ENDPOINT_URL, verbose=False,
)

print("Is server ready - {}".format(triton_http_client.is_server_ready()))

import numpy as np

def get_string_tensor(string_values, tensor_name):
    string_obj = np.array(string_values, dtype="object")
    input_obj = http_client.InferInput(tensor_name, string_obj.shape, np_to_triton_dtype(string_obj.dtype))
    input_obj.set_data_from_numpy(string_obj)
    return input_obj

def get_translation_input_for_triton(texts: list, src_lang: str, tgt_lang: str):
    return [
        get_string_tensor([[text] for text in texts], "INPUT_TEXT"),
        get_string_tensor([[src_lang]] * len(texts), "INPUT_LANGUAGE_ID"),
        get_string_tensor([[tgt_lang]] * len(texts), "OUTPUT_LANGUAGE_ID"),
    ]

inputs = get_translation_input_for_triton(["Hello world, I am Ram and I am from Ayodhya."], "en", "hi")
# inputs = get_translation_input_for_triton(["नमस्ते दुनिया, मैं रावण हूं और मैं लंकापुरी से हूं।"], "hi", "en")

# inputs = get_translation_input_for_triton(["Hello world, I am Ram and I am from Ayodhya."], "en", "mr")
# inputs = get_translation_input_for_triton(["नमस्कार दुनिया, मी रावण आहे आणि मी लंकापुरीचा आहे."], "mr", "en")

# inputs = get_translation_input_for_triton(["नमस्कार दुनिया, मी रावण आहे आणि मी लंकापुरीचा आहे."], "mr", "hi")
# inputs = get_translation_input_for_triton(["नमस्ते दुनिया, मैं रावण हूं और मैं लंकापुरी से हूं।"], "hi", "mr")

output0 = http_client.InferRequestedOutput("OUTPUT_TEXT")
response = triton_http_client.infer("nmt", model_version='1', inputs=inputs, outputs=[output0])#.get_response()

# Decode the response
output_batch = response.as_numpy('OUTPUT_TEXT').tolist()
for translation in output_batch:
    print(translation[0].decode("utf-8"))
