import math
import os
import json
import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi
import yaml
from fairseq import checkpoint_utils, tasks, utils
from fairseq.file_io import PathManager
from fairseq.data.audio.audio_utils import convert_waveform
from examples.speech_to_text.data_utils import extract_fbank_features
from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder

from simuleval.utils import entrypoint
from simuleval.data.segments import EmptySegment, TextSegment, SpeechSegment, SpeechTextSegment
from simuleval.agents import SpeechToSpeechAgent
from simuleval.agents.states import AgentStates
from simuleval.agents.actions import WriteAction, ReadAction


import pdb

SHIFT_SIZE = 10
WINDOW_SIZE = 25
SAMPLE_RATE = 16000
FEATURE_DIM = 80
BOW_PREFIX = "\u2581"
DEFAULT_EOS = 2


def _ctc_postprocess(tokens, dictionary):
    #_toks = tokens.int().tolist()
    _toks = tokens
    deduplicated_toks = [v for i, v in enumerate(_toks) if i == 0 or v != _toks[i - 1]]
    hyp = [v for v in deduplicated_toks if (v != dictionary.blank_index) and (v!= dictionary.pad_index) and (v!= dictionary.bos()) and (v!= dictionary.eos())]
    return hyp



class OnlineFeatureExtractor:
    """
    Extract speech feature on the fly.
    """

    def __init__(self, args):
        self.shift_size = args.shift_size
        self.window_size = args.window_size
        assert self.window_size >= self.shift_size

        self.original_sample_rate = args.sample_rate
        self.sample_rate = SAMPLE_RATE
        self.feature_dim = args.feature_dim
        self.num_samples_per_shift = int(self.shift_size * self.sample_rate / 1000)
        self.num_samples_per_window = int(self.window_size * self.sample_rate / 1000)
        self.len_ms_to_samples = lambda x: x * self.sample_rate / 1000
        self.previous_residual_samples = []
        self.global_cmvn = args.global_cmvn
        self.device = 'cuda' if args.device == 'gpu' else 'cpu'

    def clear_cache(self):
        self.previous_residual_samples = []

    def __call__(self, new_samples):
        samples = self.previous_residual_samples + new_samples
        if len(samples) < self.num_samples_per_window:
            self.previous_residual_samples = samples
            return

        # num_frames is the number of frames from the new segment
        num_frames = math.floor(
            (len(samples) - self.len_ms_to_samples(self.window_size - self.shift_size))
            / self.num_samples_per_shift
        )
        #TODO check num_frames pdb.set_trace()
        # the number of frames used for feature extraction
        # including some part of thte previous segment
        effective_num_samples = int(
            num_frames * self.len_ms_to_samples(self.shift_size)
            + self.len_ms_to_samples(self.window_size - self.shift_size)
        )

        input_samples = samples[:effective_num_samples]
        self.previous_residual_samples = samples[
            num_frames * self.num_samples_per_shift:
        ]

        torch.manual_seed(1)
        
        features = extract_fbank_features(torch.FloatTensor(input_samples), self.sample_rate)
        '''
        output = kaldi.fbank(
            torch.FloatTensor(input_samples).unsqueeze(0),
            num_mel_bins=self.feature_dim,
            frame_length=self.window_size,
            frame_shift=self.shift_size,
        ).numpy()
        '''
        output = self.transform(output)

        return torch.from_numpy(output, device=self.device)

    def transform(self, input):
        if self.global_cmvn is None:
            return input

        mean = self.global_cmvn["mean"]
        std = self.global_cmvn["std"]

        x = np.subtract(input, mean)
        x = np.divide(x, std)
        return x
    


class OfflineFeatureExtractor:
    """
    Extract speech feature from sequence prefix.
    """

    def __init__(self, args):
        self.shift_size = args.shift_size
        self.window_size = args.window_size
        assert self.window_size >= self.shift_size

        self.original_sample_rate = args.sample_rate
        self.sample_rate = SAMPLE_RATE
        self.feature_dim = args.feature_dim
        self.num_samples_per_shift = int(self.shift_size * self.original_sample_rate / 1000)
        self.num_samples_per_window = int(self.window_size * self.original_sample_rate / 1000)
        self.len_ms_to_samples = lambda x: x * self.original_sample_rate / 1000
        #self.previous_residual_samples = []
        self.global_cmvn = args.global_cmvn
        self.device = 'cuda' if args.device == 'gpu' else 'cpu'


    def __call__(self, new_samples):
        samples = new_samples
             
        samples, _ = convert_waveform(torch.tensor([samples]), self.original_sample_rate, to_mono=True, to_sample_rate=self.sample_rate)

        torch.manual_seed(1)
        output = extract_fbank_features(samples, self.sample_rate)
        output = self.transform(output)

        return torch.from_numpy(output).to(self.device)

    def transform(self, input):
        if self.global_cmvn is None:
            return input

        mean = self.global_cmvn["mean"]
        std = self.global_cmvn["std"]

        x = np.subtract(input, mean)
        x = np.divide(x, std)
        return x


class NASTSpeechAgentStates(AgentStates):

    def reset(self) -> None:
        """Reset Agent states"""
        
        super().reset()
        
        self.num_tgt_write = 0 
        self.last_alignment_token = None
        self.last_unit_alignment_token = None
        self.unfinished_subword = []




@entrypoint
class NASTSpeechAgent(SpeechToSpeechAgent):

    speech_segment_size = 40  # in ms, 4 pooling ratio * 10 ms step size  #TODO: confirm segment_size

    def __init__(self, args):
        super().__init__(args)
        
        self.device ='cuda' if args.device == 'gpu' else 'cpu'

        args.global_cmvn = None
        if args.config_yaml:
            with open(os.path.join(args.data_bin, args.config_yaml), "r") as f:
                config = yaml.load(f, Loader=yaml.BaseLoader)

            if "global_cmvn" in config:
                args.global_cmvn = np.load(config["global_cmvn"]["stats_npz_path"])


        self.load_model_vocab(args)
        #utils.import_user_module(args)
        
        self.feature_extractor = OfflineFeatureExtractor(args)

        self.wait_until = args.wait_until
        self.main_context = args.main_context
        self.right_context = args.right_context
        
        torch.set_grad_enabled(False)
        self.reset()
        
        with open(args.vocoder_cfg) as f:
            vocoder_cfg = json.load(f)
        self.vocoder = CodeHiFiGANVocoder(args.vocoder, vocoder_cfg)
        if args.device == 'gpu':
            self.vocoder = self.vocoder.cuda()
        self.dur_prediction = args.dur_prediction




    def build_states(self) -> NASTSpeechAgentStates:
        """
        Build states instance for agent

        Returns:
            NASTSpeechAgentStates: agent states
        """
        return NASTSpeechAgentStates()
    
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--model-path', type=str, required=True,
                            help='path to your pretrained model.')
        parser.add_argument("--data-bin", type=str, required=True,
                            help="Path of data binary")
        parser.add_argument("--config-yaml", type=str, default=None,
                            help="Path to config yaml file")
        parser.add_argument("--global-stats", type=str, default=None,
                            help="Path to json file containing cmvn stats")
        parser.add_argument("--tgt-splitter-type", type=str, default="SentencePiece",
                            help="Subword splitter type for target text")
        parser.add_argument("--tgt-splitter-path", type=str, default=None,
                            help="Subword splitter model path for target text")
        parser.add_argument("--user-dir", type=str, default="examples/simultaneous_translation",
                            help="User directory for simultaneous translation")
        parser.add_argument("--shift-size", type=int, default=SHIFT_SIZE,
                            help="Shift size of feature extraction window.")
        parser.add_argument("--window-size", type=int, default=WINDOW_SIZE,
                            help="Window size of feature extraction window.")
        parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE,
                            help="Sample rate")
        parser.add_argument("--feature-dim", type=int, default=FEATURE_DIM,
                            help="Acoustic feature dimension.")
        parser.add_argument("--main-context", type=int, default=32)
        parser.add_argument("--right-context", type=int, default=16)
        parser.add_argument("--wait-until", type=int, default=0)
        parser.add_argument("--device", type=str, default='gpu')
        parser.add_argument("--vocoder", type=str, required=True, help="path to the CodeHiFiGAN vocoder")
        parser.add_argument("--vocoder-cfg", type=str, required=True, help="path to the CodeHiFiGAN vocoder config")
        parser.add_argument("--dur-prediction", action="store_true", help="enable duration prediction (for reduced/unique code sequences)")
        parser.add_argument("--speaker-id", type=int, default=-1, help="Speaker id (for vocoder that supports multispeaker). Set to -1 to randomly sample speakers.")

        # fmt: on
        return parser

    def load_model_vocab(self, args):

        filename = args.model_path
        if not os.path.exists(filename):
            raise IOError("Model file not found: {}".format(filename))

        state = checkpoint_utils.load_checkpoint_to_cpu(filename)
        utils.import_user_module(state["cfg"].common)
        
        task_args = state["cfg"]["task"]
        task_args.data = args.data_bin

        if args.config_yaml is not None:
            task_args.config_yaml = args.config_yaml

        task = tasks.setup_task(task_args)

        # build model for ensemble
        #state["cfg"]["model"].load_pretrained_encoder_from = None
        #state["cfg"]["model"].load_pretrained_decoder_from = None
        self.model = task.build_model(state["cfg"]["model"])
        self.model.load_state_dict(state["model"], strict=True)
        self.model.eval()
        self.model.share_memory()

        if self.device == 'cuda':
            self.model.cuda()

        # Set dictionary
        self.tgt_dict = task.target_dictionary
        self.tgt_dict_unit = task.target_dictionary_unit

    @torch.inference_mode()
    def policy(self):
        
        num_frames = math.floor(
            (len(self.states.source) - self.feature_extractor.len_ms_to_samples(self.feature_extractor.window_size - self.feature_extractor.shift_size))
            / self.feature_extractor.num_samples_per_shift
        )
        #feature = self.feature_extractor(self.states.source) # prefix feature: T × C
        #pdb.set_trace()
        if self.states.source_finished:
            if num_frames == 0:
                return WriteAction(EmptySegment(finished=True), finished=True)
        else:
            if num_frames < self.main_context * (self.wait_until + 1 + self.states.num_tgt_write) + self.right_context:
                return ReadAction()              
        
        tgt_write = math.floor((num_frames - self.right_context) / self.main_context) - self.states.num_tgt_write - self.wait_until

        feature = self.feature_extractor(self.states.source) # prefix feature: T × C
        assert num_frames == feature.size(0)
        src_tokens = feature.unsqueeze(0) # 1 ×T × C
        src_lengths = torch.tensor([feature.size(0)], device=self.device).long() # 1
        
        encoder_out = self.model.forward_encoder([src_tokens, src_lengths])
        prev_decoder_out = self.model.initialize_output_tokens(encoder_out, src_tokens)
        
        if not self.states.source_finished:
            prev_decoder_out.output_tokens[-1] = self.model.unk
        
        decoder_out, unit_out = self.model.forward_streaming_decoder(prev_decoder_out, encoder_out)
        
        if self.states.source_finished:
            partial_alignments = decoder_out[int(self.main_context / 4 / self.model.unit_size) * self.states.num_tgt_write : ]
        else:
            partial_alignments = decoder_out[int(self.main_context / 4 / self.model.unit_size) * self.states.num_tgt_write : int(self.main_context / 4 / self.model.unit_size) * (self.states.num_tgt_write + tgt_write)]
        
        partial_alignments = partial_alignments.int().tolist()
        if self.states.last_alignment_token is not None and self.states.last_alignment_token != self.tgt_dict.blank_index and self.states.last_alignment_token != self.tgt_dict.bos():
            partial_alignments = [self.states.last_alignment_token] + partial_alignments
            final_output_tokens = _ctc_postprocess(partial_alignments, self.tgt_dict)[1:]
        else:
            final_output_tokens = _ctc_postprocess(partial_alignments, self.tgt_dict)
        
        self.states.last_alignment_token = partial_alignments[-1]     
        
        detok_output_tokens = []    
        
        for index in final_output_tokens:
            token = self.tgt_dict.string([index]) #return a string
            if token.startswith(BOW_PREFIX):
                if len(self.states.unfinished_subword) != 0:
                    detok_output_tokens += ["".join(self.states.unfinished_subword)]
                    self.states.unfinished_subword = []
                self.states.unfinished_subword += [token.replace(BOW_PREFIX, "")]
            else:    
                self.states.unfinished_subword += [token]
        
        if self.states.source_finished:
            detok_output_tokens += ["".join(self.states.unfinished_subword)]
            self.states.unfinished_subword = []    

        
        detok_output_string = " ".join(detok_output_tokens)
        

        #process unit output
        if self.states.source_finished:
            partial_unit_alignments = unit_out[int(self.main_context / 4 / self.model.unit_size) * self.model.hidden_upsample_ratio * self.states.num_tgt_write : ]
        else:
            partial_unit_alignments = unit_out[int(self.main_context / 4 / self.model.unit_size) * self.model.hidden_upsample_ratio * self.states.num_tgt_write : int(self.main_context / 4 / self.model.unit_size) * self.model.hidden_upsample_ratio * (self.states.num_tgt_write + tgt_write)]
        
        
        
        partial_unit_alignments = partial_unit_alignments.int().tolist()
        if self.states.last_unit_alignment_token is not None and self.states.last_unit_alignment_token != self.tgt_dict_unit.blank_index and self.states.last_unit_alignment_token != self.tgt_dict_unit.bos():
            partial_unit_alignments = [self.states.last_unit_alignment_token] + partial_unit_alignments
            final_unit_output_tokens = _ctc_postprocess(partial_unit_alignments, self.tgt_dict_unit)[1:]
        else:
            final_unit_output_tokens = _ctc_postprocess(partial_unit_alignments, self.tgt_dict_unit)
        
        self.states.last_unit_alignment_token = partial_unit_alignments[-1] 
        
        self.states.num_tgt_write += tgt_write
        if len(final_unit_output_tokens)>0:
            #pdb.set_trace()
            x = {
                "code": (torch.LongTensor(final_unit_output_tokens)-4).view(1, -1),
            }
            x = utils.move_to_cuda(x) if self.device =='cuda' else x
            wav = self.vocoder(x, self.dur_prediction).cpu().numpy().tolist()
            
            speech_out = SpeechSegment(sample_rate=16000, content=wav, tgt_lang='en', finished=self.states.source_finished)
            text_out = TextSegment(content=detok_output_string, finished=self.states.source_finished)
            #out = SpeechTextSegment(text_segment=text_out, speech_segment=speech_out)

            return WriteAction(speech_out, finished=self.states.source_finished) 
        else:
            return WriteAction(EmptySegment(finished=self.states.source_finished), finished=self.states.source_finished)
        #return WriteAction(TextSegment(content=self.tgt_dict_unit.string(final_unit_output_tokens), finished=self.states.source_finished), finished=self.states.source_finished)   
        #return WriteAction(TextSegment(content=detok_output_string, finished=self.states.source_finished), finished=self.states.source_finished) 
            
            