# Dataset Preprocessing
> [!NOTE]
> We will soon release a guide on how to use the dataset for inference and training, and how to construct these datasets on your own.

We release all the processed dataset used in our expeiments in https://huggingface.co/ICTNLP/NAST-S2X/tree/main/data


You need the following ```config.yaml``` file in the main directory of the CVSS-C data (This path should be the same to ```CVSS_ROOT``` variable in training and testing scripts.); you need to replace the placeholder with your local paths; all the files can be downloaded from this [URL](https://huggingface.co/ICTNLP/NAST-S2X/tree/main/data).

Config file:
```
bpe_tokenizer:
  bpe: sentencepiece
  sentencepiece_model: /path_to_your_data/spm_unigram10000.model
global_cmvn:
  stats_npz_path: /path_to_your_data/gcmvn.npz
input_channels: 1
input_feat_per_channel: 80
specaugment:
  freq_mask_F: 27
  freq_mask_N: 1
  time_mask_N: 1
  time_mask_T: 100
  time_mask_p: 1.0
  time_wrap_W: 0
transforms:
  '*':
  - global_cmvn
  _train:
  - global_cmvn
  - specaugment
vocab_filename: spm_unigram10000.txt
vocoder:
  checkpoint: /path_to_your_data/vocoder/mhubert_lyr11_km1000_en/g_00500000
  config: /path_to_your_data/vocoder/mhubert_lyr11_km1000_en/config.json
  type: code_hifigan

vocab_filename_src: spm_unigram10000.txt
bpe_tokenizer_src:
  bpe: sentencepiece
  sentencepiece_model: /path_to_your_data/spm_unigram10000.model
```



