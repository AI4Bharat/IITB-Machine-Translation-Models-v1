import os
import sys
import json
import numpy as np
import triton_python_backend_utils as pb_utils
from engine import Model

PWD = os.path.dirname(__file__)
CHECKPOINTS_ROOT_DIR = "/models/dhruva/checkpoints"

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        self.model_instance_device_id = json.loads(args['model_instance_device_id'])
        self.output_name = "OUTPUT_TEXT"
        self.output_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(self.model_config, self.output_name)["data_type"]
        )

        # checkpoints_root_dir = os.path.join(PWD, "checkpoints")
        checkpoints_root_dir = CHECKPOINTS_ROOT_DIR
        checkpoint_folders = [ f.path for f in os.scandir(checkpoints_root_dir) if f.is_dir() ]
        # The assumption is that, each folder name is `<src_lang>-<tgt_lang>`

        if not checkpoint_folders:
            raise RuntimeError(f"No checkpoint folders in: {checkpoints_root_dir}")

        self.models = {}
        for checkpoint_folder in checkpoint_folders:
            direction_string = os.path.basename(checkpoint_folder)
            print("Loading:", direction_string)
            src_lang, tgt_lang = direction_string.split('-')
            self.models[direction_string] = Model(checkpoint_folder+"/v1.0", src_lang, tgt_lang)
    
    def get_model(self, input_language_id, output_language_id):
        direction_string = f"{input_language_id}-{output_language_id}"
        
        if direction_string in self.models:
            return self.models[direction_string]
        raise RuntimeError(f"Language-pair not supported: {input_language_id}-{output_language_id}")

    def execute(self,requests):
        responses = []
        for request in requests:
            # TODO: Handle dynamic-batching for performance
            input_texts = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT").as_numpy()
            input_language_ids = pb_utils.get_input_tensor_by_name(request, "INPUT_LANGUAGE_ID").as_numpy()
            output_language_ids = pb_utils.get_input_tensor_by_name(request, "OUTPUT_LANGUAGE_ID").as_numpy()
            
            input_texts = [input_text[0].decode("utf-8", "ignore") for input_text in input_texts]
            input_language_ids = [input_language_id[0].decode("utf-8", "ignore") for input_language_id in input_language_ids]
            output_language_ids = [output_language_id[0].decode("utf-8", "ignore") for output_language_id in output_language_ids]

            generated_outputs = []

            # TODO: Use batch_translate()
            # translations = self.engine.batch_translate([input_text])

            for input_text, input_language_id, output_language_id in zip(input_texts, input_language_ids, output_language_ids):
                model = self.get_model(input_language_id, output_language_id)
                translation = model.translate_paragraph(input_text)
                generated_outputs.append([translation])

            inference_response = pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor(
                    self.output_name,
                    np.array(generated_outputs, dtype=self.output_dtype),
                )
            ])
            responses.append(inference_response)
        return responses
