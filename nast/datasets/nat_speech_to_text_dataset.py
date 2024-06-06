import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from dataclasses import dataclass
from fairseq.data import data_utils as fairseq_data_utils
from fairseq.data import ConcatDataset, Dictionary, FairseqDataset, ResamplingDataset
from fairseq.data.audio.dataset_transforms.concataugment import ConcatAugment
from fairseq.data.audio.speech_to_text_dataset import (
    _collate_frames,
    S2TDataConfig,
    SpeechToTextDatasetItem,
    SpeechToTextDataset,
    SpeechToTextDatasetCreator,
    TextTargetMultitaskData,
    _is_int_or_np_int,
)
from fairseq.data.audio.dataset_transforms.noisyoverlapaugment import (
    NoisyOverlapAugment,
)

logger = logging.getLogger(__name__)


@dataclass
class NATSpeechToTextDatasetItem(SpeechToTextDatasetItem):
    transcript: Optional[torch.Tensor] = None


class NATSpeechToTextDataset(SpeechToTextDataset):
    """
    Modified from SpeechToTextDataset.
    Prepend <bos> and append <eos> for target sentences for NAT generation.
    """
    def __getitem__(self, index: int) -> NATSpeechToTextDatasetItem:
        has_concat = self.dataset_transforms.has_transform(ConcatAugment)
        if has_concat:
            concat = self.dataset_transforms.get_transform(ConcatAugment)
            indices = concat.find_indices(index, self.n_frames, self.n_samples)

        source = self._get_source_audio(indices if has_concat else index)
        source = self.pack_frames(source)

        target = None
        if self.tgt_texts is not None:
            tokenized = self.get_tokenized_tgt_text(indices if has_concat else index)
            target = self.tgt_dict.encode_line(
                tokenized, add_if_not_exist=False, append_eos=True,
            ).long()
            bos = torch.LongTensor([self.tgt_dict.bos()])
            target = torch.cat((bos, target), 0)

        transcript = None
        if self.src_texts is not None:
            tokenized_transcript = self.get_tokenized_src_text(indices if has_concat else index)
            transcript = self.tgt_dict.encode_line(
                tokenized_transcript, add_if_not_exist=False, append_eos=True,
            ).long()
            bos = torch.LongTensor([self.tgt_dict.bos()])
            transcript = torch.cat((bos, transcript), 0)

        speaker_id = None
        if self.speaker_to_id is not None:
            speaker_id = self.speaker_to_id[self.speakers[index]]
        return NATSpeechToTextDatasetItem(
            index=index, source=source, transcript=transcript, target=target, speaker_id=speaker_id
        )

    def get_tokenized_src_text(self, index: Union[int, List[int]]):
        if _is_int_or_np_int(index):
            text = self.src_texts[index]
        else:
            text = " ".join([self.src_texts[i] for i in index])

        text = self.tokenize(self.pre_tokenizer, text)
        text = self.tokenize(self.bpe_tokenizer, text)
        return text
    

    def collater(
        self, samples: List[NATSpeechToTextDatasetItem], return_order: bool = False
    ) -> Dict:
        if len(samples) == 0:
            return {}
        indices = torch.tensor([x.index for x in samples], dtype=torch.long)

        sources = [x.source for x in samples]
        has_NOAug = self.dataset_transforms.has_transform(NoisyOverlapAugment)
        if has_NOAug and self.cfg.use_audio_input:
            NOAug = self.dataset_transforms.get_transform(NoisyOverlapAugment)
            sources = NOAug(sources)

        frames = _collate_frames(sources, self.cfg.use_audio_input)
        # sort samples by descending number of frames
        n_frames = torch.tensor([x.size(0) for x in sources], dtype=torch.long)
        n_frames, order = n_frames.sort(descending=True)
        indices = indices.index_select(0, order)
        frames = frames.index_select(0, order)

        target, target_lengths = None, None
        ntokens = None
        if self.tgt_texts is not None:
            target = fairseq_data_utils.collate_tokens(
                [x.target for x in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            target = target.index_select(0, order)
            target_lengths = torch.tensor(
                [x.target.size(0) for x in samples], dtype=torch.long
            ).index_select(0, order)
            ntokens = sum(x.target.size(0) for x in samples)

        transcript, transcript_lengths = None, None
        ntokens_transcript = None
        if self.src_texts is not None:
            transcript = fairseq_data_utils.collate_tokens(
                [x.transcript for x in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            transcript = transcript.index_select(0, order)
            transcript_lengths = torch.tensor(
                [x.transcript.size(0) for x in samples], dtype=torch.long
            ).index_select(0, order)
            ntokens_transcript = sum(x.transcript.size(0) for x in samples)
        
        speaker = None
        if self.speaker_to_id is not None:
            speaker = (
                torch.tensor([s.speaker_id for s in samples], dtype=torch.long)
                .index_select(0, order)
                .view(-1, 1)
            )

        net_input = {
            "src_tokens": frames,
            "src_lengths": n_frames,
        }
        out = {
            "id": indices,
            "net_input": net_input,
            "speaker": speaker,
            "target": target,
            "target_lengths": target_lengths,
            "transcript": transcript,
            "transcript_lengths": transcript_lengths,
            "ntokens": ntokens,
            "nsentences": len(samples),
        }
        if return_order:
            out["order"] = order
        return out


class NATTextTargetMultitaskData(TextTargetMultitaskData):
    """
    Modified from TextTargetMultitaskData.
    Prepend <bos> and append <eos> for target sentences for NAT generation.
    """
    def get(self, sample_id, tgt_lang=None):
        if sample_id in self.data:
            tokenized = self.get_tokenized_tgt_text(sample_id)
            target = self.dict.encode_line(
                tokenized,
                add_if_not_exist=False,
                append_eos=True,
            )
            bos = torch.LongTensor([self.dict.bos()])
            target = torch.cat((bos, target), 0)
            return target
        else:
            logger.warning(f"no target for {sample_id}")
            return torch.IntTensor([])

    def collater(self, samples: List[torch.Tensor]) -> torch.Tensor:
        out = fairseq_data_utils.collate_tokens(
            samples,
            self.dict.pad(),
            eos_idx=None,
            left_pad=False,
            move_eos_to_beginning=False,
        ).long()

        target_lengths = torch.tensor([t.size(0) for t in samples], dtype=torch.long)
        ntokens = sum(t.size(0) for t in samples)

        output = {
            "target": out,
            "target_lengths": target_lengths,
            "ntokens": ntokens,
        }

        return output


class NATSpeechToTextMultitaskDataset(NATSpeechToTextDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.multitask_data = {}

    def add_multitask_dataset(self, task_name, task_data):
        self.multitask_data[task_name] = task_data

    def __getitem__(
        self, index: int
    ) -> Tuple[SpeechToTextDatasetItem, Dict[str, torch.Tensor]]:
        s2t_data = super().__getitem__(index)

        multitask_target = {}
        sample_id = self.ids[index]
        tgt_lang = self.tgt_langs[index]
        for task_name, task_dataset in self.multitask_data.items():
            multitask_target[task_name] = task_dataset.get(sample_id, tgt_lang)

        return s2t_data, multitask_target

    def collater(
        self, samples: List[Tuple[SpeechToTextDatasetItem, Dict[str, torch.Tensor]]]
    ) -> Dict:
        if len(samples) == 0:
            return {}

        out = super().collater([s for s, _ in samples], return_order=True)
        order = out["order"]
        del out["order"]

        for task_name, task_dataset in self.multitask_data.items():
            if "multitask" not in out:
                out["multitask"] = {}
            d = [s[task_name] for _, s in samples]
            task_target = task_dataset.collater(d)
            out["multitask"][task_name] = {
                "target": task_target["target"].index_select(0, order),
                "target_lengths": task_target["target_lengths"].index_select(0, order),
                "ntokens": task_target["ntokens"],
            }

        return out


class NATSpeechToTextDatasetCreator(SpeechToTextDatasetCreator):
    DEFAULT_TGT_TEXT = ""

    @classmethod
    def _from_list(
        cls,
        split_name: str,
        is_train_split,
        samples: List[Dict],
        cfg: S2TDataConfig,
        tgt_dict,
        pre_tokenizer,
        bpe_tokenizer,
        n_frames_per_step,
        speaker_to_id,
        multitask: Optional[Dict] = None,
    ) -> NATSpeechToTextDataset:
        audio_root = Path(cfg.audio_root)
        ids = [s[cls.KEY_ID] for s in samples]
        audio_paths = [(audio_root / s[cls.KEY_AUDIO]).as_posix() for s in samples]
        n_frames = [int(s[cls.KEY_N_FRAMES]) for s in samples]
        tgt_texts = [s.get(cls.KEY_TGT_TEXT, cls.DEFAULT_TGT_TEXT) for s in samples]
        src_texts = [s.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) for s in samples]
        speakers = [s.get(cls.KEY_SPEAKER, cls.DEFAULT_SPEAKER) for s in samples]
        src_langs = [s.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for s in samples]
        tgt_langs = [s.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for s in samples]

        has_multitask = multitask is not None and len(multitask.keys()) > 0
        dataset_cls = (
            NATSpeechToTextMultitaskDataset if has_multitask else NATSpeechToTextDataset
        )

        ds = dataset_cls(
            split=split_name,
            is_train_split=is_train_split,
            cfg=cfg,
            audio_paths=audio_paths,
            n_frames=n_frames,
            src_texts=src_texts,
            tgt_texts=tgt_texts,
            speakers=speakers,
            src_langs=src_langs,
            tgt_langs=tgt_langs,
            ids=ids,
            tgt_dict=tgt_dict,
            pre_tokenizer=pre_tokenizer,
            bpe_tokenizer=bpe_tokenizer,
            n_frames_per_step=n_frames_per_step,
            speaker_to_id=speaker_to_id,
        )

        if has_multitask:
            for task_name, task_obj in multitask.items():
                task_data = NATTextTargetMultitaskData(
                    task_obj.args, split_name, task_obj.target_dictionary
                )
                ds.add_multitask_dataset(task_name, task_data)
        return ds

    @classmethod
    def _from_tsv(
        cls,
        root: str,
        cfg: S2TDataConfig,
        split: str,
        tgt_dict,
        is_train_split: bool,
        pre_tokenizer,
        bpe_tokenizer,
        n_frames_per_step,
        speaker_to_id,
        multitask: Optional[Dict] = None,
    ) -> NATSpeechToTextDataset:
        samples = cls._load_samples_from_tsv(root, split)
        return cls._from_list(
            split,
            is_train_split,
            samples,
            cfg,
            tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            n_frames_per_step,
            speaker_to_id,
            multitask,
        )

    @classmethod
    def from_tsv(
        cls,
        root: str,
        cfg: S2TDataConfig,
        splits: str,
        tgt_dict,
        pre_tokenizer,
        bpe_tokenizer,
        is_train_split: bool,
        epoch: int,
        seed: int,
        n_frames_per_step: int = 1,
        speaker_to_id=None,
        multitask: Optional[Dict] = None,
    ) -> NATSpeechToTextDataset:
        datasets = [
            cls._from_tsv(
                root=root,
                cfg=cfg,
                split=split,
                tgt_dict=tgt_dict,
                is_train_split=is_train_split,
                pre_tokenizer=pre_tokenizer,
                bpe_tokenizer=bpe_tokenizer,
                n_frames_per_step=n_frames_per_step,
                speaker_to_id=speaker_to_id,
                multitask=multitask,
            )
            for split in splits.split(",")
        ]

        if is_train_split and len(datasets) > 1 and cfg.sampling_alpha != 1.0:
            # temperature-based sampling
            size_ratios = cls.get_size_ratios(datasets, alpha=cfg.sampling_alpha)
            datasets = [
                ResamplingDataset(
                    d, size_ratio=r, seed=seed, epoch=epoch, replace=(r >= 1.0)
                )
                for r, d in zip(size_ratios, datasets)
            ]

        return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]