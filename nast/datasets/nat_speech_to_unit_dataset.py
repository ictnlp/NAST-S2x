import torch
import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from fairseq.data import data_utils as fairseq_data_utils
from fairseq.data import ConcatDataset, Dictionary, FairseqDataset
from fairseq.data.audio.speech_to_text_dataset import (
    _collate_frames,
    _is_int_or_np_int,
    SpeechToTextDatasetCreator,
)
from fairseq.data.audio.audio_utils import get_features_or_waveform
from fairseq.data.audio.feature_transforms import CompositeAudioFeatureTransform
from fairseq.data.audio.waveform_transforms import CompositeAudioWaveformTransform
from fairseq.data.audio.data_cfg import S2SDataConfig

logger = logging.getLogger(__name__)


class S2UDataConfig(S2SDataConfig):
    """Wrapper class for data config YAML"""
    @property
    def pre_tokenizer(self) -> Dict:
        """Pre-tokenizer to apply before subword tokenization. Returning
        a dictionary with `tokenizer` providing the tokenizer name and
        the other items providing the tokenizer-specific arguments.
        Tokenizers are defined in `fairseq.data.encoders.*`"""
        tokenizer = self.config.get("pre_tokenizer", {"tokenizer": None})
        return self._auto_convert_to_abs_path(tokenizer)

    @property
    def bpe_tokenizer(self) -> Dict:
        """Subword tokenizer to apply after pre-tokenization. Returning
        a dictionary with `bpe` providing the tokenizer name and
        the other items providing the tokenizer-specific arguments.
        Tokenizers are defined in `fairseq.data.encoders.*`"""
        tokenizer = self.config.get("bpe_tokenizer", {"bpe": None})
        return self._auto_convert_to_abs_path(tokenizer)





@dataclass
class NATSpeechToUnitDatasetItem(object):
    index: int
    source: torch.Tensor
    source_text: Optional[torch.Tensor] = None
    target_text: Optional[torch.Tensor] = None
    target_audio: Optional[torch.Tensor] = None

    
class NATSpeechToUnitDataset(FairseqDataset):
    def __init__(
        self,
        split: str,
        is_train_split: bool,
        cfg: S2UDataConfig,
        src_audio_paths: List[str],
        src_n_frames: List[int],
        src_texts: Optional[List[str]] = None,
        tgt_audio_paths: Optional[List[str]] = None,
        tgt_n_frames: Optional[List[int]] = None,
        tgt_texts: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
        tgt_dict: Optional[Dictionary] = None,
        tgt_dict_unit: Optional[Dictionary] = None,
        pre_tokenizer=None,
        bpe_tokenizer=None,
        n_frames_per_step: int = 1,
    ):
        self.split, self.is_train_split = split, is_train_split
        self.cfg = cfg
        self.src_audio_paths, self.src_n_frames = src_audio_paths, src_n_frames
        self.tgt_audio_paths, self.tgt_n_frames = tgt_audio_paths, tgt_n_frames
        self.n_samples = len(src_audio_paths)
        assert len(src_n_frames) == self.n_samples > 0
        self.src_texts, self.tgt_texts = src_texts, tgt_texts
        self.tgt_dict = tgt_dict
        self.tgt_dict_unit = tgt_dict_unit
        self.ids = ids
        self.shuffle = cfg.shuffle if is_train_split else False
        
        self.feature_transforms = CompositeAudioFeatureTransform.from_config_dict(
            self.cfg.get_feature_transforms(split, is_train_split)
        )
        self.waveform_transforms = CompositeAudioWaveformTransform.from_config_dict(
            self.cfg.get_waveform_transforms(split, is_train_split)
        )

        # NOTE: currently not support raw audio input
        assert not self.cfg.use_audio_input
        
        
        self.pre_tokenizer = pre_tokenizer
        self.bpe_tokenizer = bpe_tokenizer
        self.tgt_lens = self.get_tgt_lens_and_check_oov()

        # NOTE: n_frames_per_step is used for target audio rather than source audio
        self.n_frames_per_step = n_frames_per_step  

        logger.info(self.__repr__())

    def get_tgt_lens_and_check_oov(self):
        if self.tgt_texts is None:
            return [0 for _ in range(self.n_samples)]
        tgt_lens = []
        n_tokens, n_oov_tokens = 0, 0
        for i in range(self.n_samples):
            tokenized = self.get_tokenized_tgt_text(i).split(" ")
            oov_tokens = [
                t
                for t in tokenized
                if self.tgt_dict.index(t) == self.tgt_dict.unk_index
            ]
            n_tokens += len(tokenized)
            n_oov_tokens += len(oov_tokens)
            tgt_lens.append(len(tokenized))
        logger.info(f"'{self.split}' has {n_oov_tokens / n_tokens * 100:.2f}% OOV")
        return tgt_lens

    def __repr__(self):
        return (
            self.__class__.__name__
            + f'(split="{self.split}", n_samples={self.n_samples:_}, '
            f"prepend_tgt_lang_tag={self.cfg.prepend_tgt_lang_tag}, "
            f"n_frames_per_step={self.n_frames_per_step}, "
            f"shuffle={self.shuffle}, "
            f"feature_transforms={self.feature_transforms}, "
            f"waveform_transforms={self.waveform_transforms}"
        )
    
    
    @classmethod
    def tokenize(cls, tokenizer, text: str):
        return text if tokenizer is None else tokenizer.encode(text)

    def get_tokenized_tgt_text(self, index: Union[int, List[int]]):
        if _is_int_or_np_int(index):
            text = self.tgt_texts[index]
        else:
            text = " ".join([self.tgt_texts[i] for i in index])

        text = self.tokenize(self.pre_tokenizer, text)
        text = self.tokenize(self.bpe_tokenizer, text)
        return text
    
    def get_tokenized_src_text(self, index: Union[int, List[int]]):
        if _is_int_or_np_int(index):
            text = self.src_texts[index]
        else:
            text = " ".join([self.src_texts[i] for i in index])

        text = self.tokenize(self.pre_tokenizer, text)
        text = self.tokenize(self.bpe_tokenizer, text)
        return text

    def _get_source_audio(self, index: int) -> torch.Tensor:
        """
        Gives source audio for given index with any relevant transforms applied.
        """
        source = get_features_or_waveform(
            self.src_audio_paths[index],
            waveform_transforms=self.waveform_transforms,
        )
        if self.feature_transforms is not None:
            source = self.feature_transforms(source)
        source = torch.from_numpy(source).float()
        return source

    def __getitem__(self, index: int) -> NATSpeechToUnitDatasetItem:
        # source audio
        source = self._get_source_audio(index)

        # target text
        # NOTE: append eos and prepend bos
        target_text = None
        if self.tgt_texts is not None:
            tokenized = self.get_tokenized_tgt_text(index)
            target_text = self.tgt_dict.encode_line(
                tokenized, add_if_not_exist=False, append_eos=True,
            ).long()
            bos = torch.LongTensor([self.tgt_dict.bos()])
            target_text = torch.cat((bos, target_text), 0)

        source_text = None
        if self.src_texts is not None:
            tokenized_source_text = self.get_tokenized_src_text(index)
            source_text = self.tgt_dict.encode_line(
                tokenized_source_text, add_if_not_exist=False, append_eos=True,
            ).long()
            bos = torch.LongTensor([self.tgt_dict.bos()])
            source_text = torch.cat((bos, source_text), 0)
        
        # target audio
        target_audio = None
        if self.tgt_audio_paths is not None:
            target_audio = self.tgt_dict_unit.encode_line(
                self.tgt_audio_paths[index],
                add_if_not_exist=False,
                append_eos=True,
            ).long()
            bos = torch.LongTensor([self.tgt_dict_unit.bos()])
            target_audio = torch.cat((bos, target_audio), 0)

        return NATSpeechToUnitDatasetItem(
            index=index, source=source, target_text=target_text, source_text=source_text, target_audio=target_audio,
        )

    def collater(
        self, samples: List[NATSpeechToUnitDatasetItem], return_order: bool = False
    ) -> Dict:
        if len(samples) == 0:
            return {}
        indices = torch.tensor([x.index for x in samples], dtype=torch.long)
        # source audio
        sources = [x.source for x in samples]
        frames = _collate_frames(sources, self.cfg.use_audio_input)
        # sort samples by descending number of frames
        n_frames = torch.tensor([x.size(0) for x in sources], dtype=torch.long)
        n_frames, order = n_frames.sort(descending=True)
        indices = indices.index_select(0, order)
        frames = frames.index_select(0, order)

        # target text
        target_text, target_text_lengths, ntokens_text = None, None, None
        if self.tgt_texts is not None:
            target_text = fairseq_data_utils.collate_tokens(
                [x.target_text for x in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            ).index_select(0, order)
            target_text_lengths = torch.tensor(
                [x.target_text.size(0) for x in samples], dtype=torch.long
            ).index_select(0, order)
            ntokens_text = sum(x.target_text.size(0) for x in samples)
        
        # source text
        source_text, source_text_lengths = None, None
        if self.src_texts is not None:
            source_text = fairseq_data_utils.collate_tokens(
                [x.source_text for x in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            ).index_select(0, order)
            source_text_lengths = torch.tensor(
                [x.source_text.size(0) for x in samples], dtype=torch.long
            ).index_select(0, order)
            ntokens_text = sum(x.source_text.size(0) for x in samples)

        # target audio
        target_audio, target_audio_lengths, ntokens_audio = None, None, None
        if self.tgt_audio_paths is not None:
            target_audio = fairseq_data_utils.collate_tokens(
                [x.target_audio for x in samples],
                self.tgt_dict_unit.pad(),
                self.tgt_dict_unit.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            ).index_select(0, order)
            target_audio_lengths = torch.tensor(
                [x.target_audio.size(0) for x in samples], dtype=torch.long
            ).index_select(0, order)
            ntokens_audio = sum(x.target_audio.size(0) for x in samples)

        net_input = {
            "src_tokens": frames,
            "src_lengths": n_frames,
        }
        out = {
            "id": indices,
            "net_input": net_input,
            "source_text": source_text,
            "source_text_lengths": source_text_lengths,
            "target_text": target_text,
            "target_text_lengths": target_text_lengths,
            "target_audio": target_audio,
            "target_audio_lengths": target_audio_lengths,
            "ntokens_text": ntokens_text,
            "ntokens_audio": ntokens_audio,
            "nsentences": len(samples),
        }
        if return_order:
            out["order"] = order
        return out

    def __len__(self):
        return self.n_samples

    def num_tokens(self, index):
        return self.src_n_frames[index]
    
    def size(self, index):
        return self.src_n_frames[index], self.tgt_lens[index], self.tgt_n_frames[index]
    
    @property
    def sizes(self):
        return np.array(self.src_n_frames)
    
    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        # first by descending order of # of frames then by original/random order
        order.append([-n for n in self.src_n_frames])
        return np.lexsort(order)

    def prefetch(self, indices):
        raise False



class NATSpeechToUnitDatasetCreator(object):
    # mandatory columns
    KEY_ID, KEY_SRC_AUDIO, KEY_SRC_N_FRAMES = "id", "audio", "n_frames"
    KEY_TGT_AUDIO, KEY_TGT_N_FRAMES = "tgt_audio", "tgt_n_frames"
    # optional columns
    KEY_TGT_TEXT = "tgt_text"
    KEY_SRC_TEXT = "src_text"

    @classmethod
    def _from_list(
        cls,
        split_name: str,
        is_train_split,
        samples: List[Dict],
        cfg: S2UDataConfig,
        tgt_dict: Dictionary = None,
        tgt_dict_unit: Dictionary = None,
        pre_tokenizer = None,
        bpe_tokenizer = None,
        n_frames_per_step: int = 1,
    ) -> NATSpeechToUnitDataset:
        ids = [s[cls.KEY_ID] for s in samples]
        src_audio_paths = [s[cls.KEY_SRC_AUDIO] for s in samples]
        src_n_frames = [int(s[cls.KEY_SRC_N_FRAMES]) for s in samples]
        tgt_audio_paths = [s.get(cls.KEY_TGT_AUDIO, None) for s in samples]
        tgt_n_frames = [int(s.get(cls.KEY_TGT_N_FRAMES, 0)) for s in samples]
        tgt_texts = [s.get(cls.KEY_TGT_TEXT, "") for s in samples]
        src_texts = [s.get(cls.KEY_SRC_TEXT, "") for s in samples]

        tgt_audio_paths = None if any(tgt is None for tgt in tgt_audio_paths) else tgt_audio_paths

        ds = NATSpeechToUnitDataset(
            split=split_name,
            is_train_split=is_train_split,
            cfg=cfg,
            src_audio_paths=src_audio_paths,
            src_n_frames=src_n_frames,
            src_texts=src_texts,
            tgt_audio_paths=tgt_audio_paths,
            tgt_n_frames=tgt_n_frames,
            tgt_texts=tgt_texts,
            ids=ids,
            tgt_dict=tgt_dict,
            tgt_dict_unit=tgt_dict_unit,
            pre_tokenizer=pre_tokenizer,
            bpe_tokenizer=bpe_tokenizer,
            n_frames_per_step=n_frames_per_step,
        )

        return ds

    @classmethod
    def from_tsv(
        cls,
        root: str,
        cfg: S2UDataConfig,
        splits: str,
        is_train_split: bool,
        tgt_dict: Dictionary = None,
        tgt_dict_unit: Dictionary = None,
        pre_tokenizer = None,
        bpe_tokenizer = None,
        n_frames_per_step: int = 1,
    ) -> NATSpeechToUnitDataset:
        datasets = []
        for split in splits.split(","):
            samples = SpeechToTextDatasetCreator._load_samples_from_tsv(root, split)
            ds = cls._from_list(
                split_name=split,
                is_train_split=is_train_split,
                samples=samples,
                cfg=cfg,
                tgt_dict=tgt_dict,
                tgt_dict_unit=tgt_dict_unit,
                pre_tokenizer=pre_tokenizer,
                bpe_tokenizer=bpe_tokenizer,
                n_frames_per_step=n_frames_per_step,
            )
            datasets.append(ds)
        return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]