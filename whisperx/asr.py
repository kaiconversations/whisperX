"""
Essentially added additional inference method which uses the faster-whisper batched inference pipeline instead of
WhisperX pipeline which doesn't work for arabic-english code switching.

For use with experimental version of faster-whisper https://github.com/SYSTRAN/faster-whisper/pull/936
Run "pip install git+https://github.com/MahmoudAshraf97/faster-whisper.git@same_vad" to install
This version doesn't change the WhisperModel class which is used by whisperX for the legacy inference method, but does
provide the faster-whisper.BatchedInferencePipeline so for now it's our best bet.

This has been designed such that minial changes are required to the transcription engine as both pipelines use the same
function names.
"""
import inspect
import logging
import os
from typing import Optional, List, Union, Tuple, Iterable

import ctranslate2
import numpy as np
import faster_whisper
import torch
from faster_whisper.transcribe import TranscriptionOptions, get_suppressed_tokens

from transformers import Pipeline
from transformers.pipelines.pt_utils import PipelineIterator

from .audio import N_SAMPLES, SAMPLE_RATE, log_mel_spectrogram, load_audio
from .types import TranscriptionResult, SingleSegment
from .vad import merge_chunks, load_vad_model


def find_numeral_symbol_tokens(tokenizer):
    numeral_symbol_tokens = []
    for i in range(tokenizer.eot):
        token = tokenizer.decode([i]).removeprefix(" ")
        has_numeral_symbol = any(c in "0123456789%$£" for c in token)
        if has_numeral_symbol:
            numeral_symbol_tokens.append(i)
    return numeral_symbol_tokens


class WhisperXModel(faster_whisper.WhisperModel):
    """
    FasterWhisperModel provides batched inference for faster-whisper.
    Currently only works in non-timestamp mode and fixed prompt for all samples in batch.
    """

    def generate_segment_batched(
        self,
        features: np.ndarray,
        tokenizer: faster_whisper.tokenizer.Tokenizer,
        options: faster_whisper.transcribe.TranscriptionOptions,
        encoder_output=None
    ):
        batch_size = features.shape[0]
        all_tokens = []
        prompt_reset_since = 0
        if options.initial_prompt is not None:
            initial_prompt = " " + options.initial_prompt.strip()
            initial_prompt_tokens = tokenizer.encode(initial_prompt)
            all_tokens.extend(initial_prompt_tokens)
        previous_tokens = all_tokens[prompt_reset_since:]
        prompt = self.get_prompt(
            tokenizer,
            previous_tokens,
            without_timestamps=options.without_timestamps,
            prefix=options.prefix,
        )

        encoder_output = self.encode(features)

        max_initial_timestamp_index = int(
            round(options.max_initial_timestamp / self.time_precision)
        )

        result = self.model.generate(
            encoder_output,
            [prompt] * batch_size,
            beam_size=options.beam_size,
            patience=options.patience,
            length_penalty=options.length_penalty,
            max_length=self.max_length,
            suppress_blank=options.suppress_blank,
            suppress_tokens=options.suppress_tokens,
        )

        tokens_batch = [x.sequences_ids[0] for x in result]

        def decode_batch(tokens: List[List[int]]) -> str:
            res = []
            for tk in tokens:
                res.append([token for token in tk if token < tokenizer.eot])
            # text_tokens = [token for token in tokens if token < self.eot]
            return tokenizer.tokenizer.decode_batch(res)

        text = decode_batch(tokens_batch)

        return text

    def encode(self, features: np.ndarray) -> ctranslate2.StorageView:
        # When the model is running on multiple GPUs, the encoder output should be moved
        # to the CPU since we don't know which GPU will handle the next job.
        to_cpu = self.model.device == "cuda" and len(self.model.device_index) > 1
        # unsqueeze if batch size = 1
        if len(features.shape) == 2:
            features = np.expand_dims(features, 0)
        features = faster_whisper.transcribe.get_ctranslate2_storage(features)

        return self.model.encode(features, to_cpu=to_cpu)


class WhisperXPipeline(Pipeline):
    def __init__(
            self,
            model,
            vad,
            vad_params: dict,
            tokenizer=None,
            device: Union[int, str, "torch.device"] = -1,
            framework="pt",
            language: Optional[str] = None,
            **kwargs
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.preset_language = language
        self.options = None
        self._preprocess_params, self._forward_params, self._postprocess_params = self._sanitize_parameters(**kwargs)
        self.call_count = 0
        self.framework = framework
        if self.framework == "pt":
            if isinstance(device, torch.device):
                self.device = device
            elif isinstance(device, str):
                self.device = torch.device(device)
            elif device < 0:
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(f"cuda:{device}")
        else:
            self.device = device

        super(Pipeline, self).__init__()
        self.vad_model = vad
        self._vad_params = vad_params

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "tokenizer" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, audio):
        audio = audio['inputs']
        model_n_mels = self.model.feat_kwargs.get("feature_size")
        features = log_mel_spectrogram(
            audio,
            n_mels=model_n_mels if model_n_mels is not None else 80,
            padding=N_SAMPLES - audio.shape[0],
        )
        return {'inputs': features}

    def _forward(self, model_inputs):
        outputs = self.model.generate_segment_batched(model_inputs['inputs'], self.tokenizer, self.options)
        return {'text': outputs}

    def postprocess(self, model_outputs):
        return model_outputs

    def get_iterator(
            self, inputs, num_workers: int, batch_size: int, preprocess_params, forward_params, postprocess_params
    ):
        dataset = PipelineIterator(inputs, self.preprocess, preprocess_params)
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # TODO hack by collating feature_extractor and image_processor

        def stack(items):
            return {'inputs': torch.stack([x['inputs'] for x in items])}
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=stack)
        model_iterator = PipelineIterator(dataloader, self.forward, forward_params, loader_batch_size=batch_size)
        final_iterator = PipelineIterator(model_iterator, self.postprocess, postprocess_params)
        return final_iterator

    def transcribe(
            self,
            audio: np.ndarray,
            num_workers=0,
            language=None,
            task=None,
            chunk_size=30,
            log_progress=False,
            beam_size: int = 5,
            best_of: int = 5,
            patience: float = 1,
            length_penalty: float = 1,
            repetition_penalty: float = 1,
            no_repeat_ngram_size: int = 0,
            temperature: Union[float, List[float], Tuple[float, ...]] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            compression_ratio_threshold: Optional[float] = 2.4,
            log_prob_threshold: Optional[float] = -1.0,
            log_prob_low_threshold: Optional[float] = None,
            no_speech_threshold: Optional[float] = 0.6,
            initial_prompt: Optional[Union[str, Iterable[int]]] = None,
            prefix: Optional[str] = None,
            suppress_blank: bool = True,
            suppress_tokens: Optional[List[int]] = [-1],
            without_timestamps: bool = True,
            word_timestamps: bool = False,
            prepend_punctuations: str = "\"'“¿([{-",
            append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
            max_new_tokens: Optional[int] = None,
            batch_size: int = 16,
            hotwords: Optional[str] = None,
            **kwargs,
    ) -> TranscriptionResult:
        if isinstance(audio, str):
            audio = load_audio(audio)

        def data(audio, segments):
            for seg in segments:
                f1 = int(seg['start'] * SAMPLE_RATE)
                f2 = int(seg['end'] * SAMPLE_RATE)
                # print(f2-f1)
                yield {'inputs': audio[f1:f2]}

        vad_segments = self.vad_model({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SAMPLE_RATE})
        vad_segments = merge_chunks(
            vad_segments,
            chunk_size,
            onset=self._vad_params["vad_onset"],
            offset=self._vad_params["vad_offset"],
        )
        if self.tokenizer is None:
            language = language or self.detect_language(audio)
            task = task or "transcribe"
            self.tokenizer = faster_whisper.tokenizer.Tokenizer(
                self.model.hf_tokenizer,
                self.model.model.is_multilingual,
                task=task,
                language=language
            )
        else:
            language = language or self.tokenizer.language_code
            task = task or self.tokenizer.task
            if task != self.tokenizer.task or language != self.tokenizer.language_code:
                self.tokenizer = faster_whisper.tokenizer.Tokenizer(
                    self.model.hf_tokenizer,
                    self.model.model.is_multilingual,
                    task=task,
                    language=language
                )

        options = TranscriptionOptions(
            beam_size=beam_size,
            best_of=best_of,
            patience=patience,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            log_prob_threshold=log_prob_threshold,
            log_prob_low_threshold=log_prob_low_threshold,
            no_speech_threshold=no_speech_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            temperatures=(
                temperature if isinstance(temperature, (list, tuple)) else [temperature]
            ),
            initial_prompt=initial_prompt,
            prefix=prefix,
            suppress_blank=suppress_blank,
            suppress_tokens=get_suppressed_tokens(self.tokenizer, suppress_tokens),
            prepend_punctuations=prepend_punctuations,
            append_punctuations=append_punctuations,
            max_new_tokens=max_new_tokens,
            hotwords=hotwords,
            word_timestamps=word_timestamps,
            hallucination_silence_threshold=None,
            condition_on_previous_text=False,
            clip_timestamps="0",
            prompt_reset_on_temperature=0.5,
            multilingual=False,
            output_language=None,
            without_timestamps=without_timestamps,
            max_initial_timestamp=0.0,
        )

        self.options = options

        self.options = options

        segments: List[SingleSegment] = []
        total_segments = len(vad_segments)
        for idx, out in enumerate(self.__call__(data(audio, vad_segments), batch_size=batch_size, num_workers=num_workers)):
            if log_progress:
                base_progress = ((idx + 1) / total_segments) * 100
                percent_complete = base_progress / 2
                print(f"Progress: {percent_complete:.2f}%...")
            text = out['text']
            if batch_size in [0, 1, None]:
                text = text[0]
            segments.append(
                {
                    "text": text,
                    "start": round(vad_segments[idx]['start'], 3),
                    "end": round(vad_segments[idx]['end'], 3)
                }
            )

        # revert the tokenizer if multilingual inference is enabled
        if self.preset_language is None:
            self.tokenizer = None

        return {"segments": segments, "language": language}

    def detect_language(self, audio: np.ndarray):
        if audio.shape[0] < N_SAMPLES:
            print("Warning: audio is shorter than 30s, language detection may be inaccurate.")
        model_n_mels = self.model.feat_kwargs.get("feature_size")
        segment = log_mel_spectrogram(audio[: N_SAMPLES],
                                      n_mels=model_n_mels if model_n_mels is not None else 80,
                                      padding=0 if audio.shape[0] >= N_SAMPLES else N_SAMPLES - audio.shape[0])
        encoder_output = self.model.encode(segment)
        results = self.model.model.detect_language(encoder_output)
        language_token, language_probability = results[0][0]
        language = language_token[2:-2]
        print(f"Detected language: {language} ({language_probability:.2f}) in first 30s of audio...")
        return language


class BatchedFasterWhisperPipeline(faster_whisper.BatchedInferencePipeline):
    def __init__(
            self,
            whisper_model: faster_whisper.WhisperModel,
            language: Optional[str] = None,
    ):
        super().__init__(whisper_model, language=language)
        transcribe_signature = inspect.signature(super().transcribe)
        self.valid_params = transcribe_signature.parameters.keys()
        logging.info(f"BatchedFasterWhisperPipeline valid params: {self.valid_params}")

    def transcribe(self, audio: np.ndarray, **kwargs):
        logging.info(f"BatchedFasterWhisperPipeline transcribe kwargs: {kwargs}")
        valid_kwargs = {k: v for k, v in kwargs.items() if k in self.valid_params}
        logging.info(f"BatchedFasterWhisperPipeline transcribe VALID kwargs: {valid_kwargs}")
        types = {k: type(v) for k, v in valid_kwargs.items()}
        logging.info(f"kwargs Types: {types}")

        segments, info = super().transcribe(audio=audio, **valid_kwargs)
        whisperx_segments = []
        for segment in segments:
            whisperx_segments.append(SingleSegment(
                text=segment.text,
                start=segment.start,
                end=segment.end
            ))

        return TranscriptionResult(
            segments=whisperx_segments,
            language=self.preset_language,
        )


def load_model(
        model_name: str,
        inference_type: str = "legacy",
        device: str = "cuda",
        device_index: int = 0,
        compute_type: str = "float16",
        cpu_threads: int = 4,
        num_workers: int = 1,
        task: str = "transcribe",
        language: str = None,
        download_root: Optional[str] = None,
):
    """Load a Whisper model for inference.

    Args:
        model_name (str): The model name to load.
        inference_type (str): WhisperX implementation = "legacy" or experimental faster-whisper "experimental"
        device (str): The device to use for inference.
        compute_type (str): The compute type to use for inference.
        num_workers (int): The number of workers to use for inference.
        language (str): The language to use for inference.
        download_root (str): The root directory to download models to.
    """

    if inference_type == "experimental":
        model = faster_whisper.WhisperModel(
            model_name,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            num_workers=num_workers,
            download_root=download_root
        )
        return BatchedFasterWhisperPipeline(
            model,
            language=language
        )
    else:
        model = WhisperXModel(
            model_name,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            download_root=download_root,
            cpu_threads=cpu_threads
        )

        tokenizer = faster_whisper.tokenizer.Tokenizer(model.hf_tokenizer, model.model.is_multilingual, task=task, language=language)
        default_vad_options = {
            "vad_onset": 0.500,
            "vad_offset": 0.363
        }
        vad_model = load_vad_model(torch.device(device), use_auth_token=None, **default_vad_options)

        return WhisperXPipeline(
            model,
            vad=vad_model,
            vad_params=default_vad_options,
            tokenizer=tokenizer,
            language=language,
        )
