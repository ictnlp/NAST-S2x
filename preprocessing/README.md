# Dataset Preprocessing
> [!NOTE]
> We release all the processed dataset used in our expeiments in https://huggingface.co/ICTNLP/NAST-S2X/tree/main/data.




## How to use the datasets?
**You need to organize your data as follows:**

```
$CVSS_ROOT
└── fr-en
    ├── gcmvn.npz            
    ├── src_fbank80.zip
    └── fbank2unit
    |        └── config.yaml
    |        ├── train.tsv
    |        ├── test.tsv
    |        ├── dev.tsv
    |        ├── spm_unigram10000.model
    |        ├── spm_unigram10000.txt             
    |        └── spm_unigram10000.vocab
    └── vocoder
             └── mhubert_lyr11_km1000_en
                              ├── config.json 
                              └── g_00500000
```
```$CVSS_ROOT``` should match the paths used in the training and testing scripts.


All files can be downloaded from this [URL](https://huggingface.co/ICTNLP/NAST-S2X/tree/main/data) except for ```config.yaml``` and ```src_fbank80.zip```. 

We have provided a sample ```config.yaml``` file below. You may need to modify some of the path according to your situation.
```
bpe_tokenizer:
  bpe: sentencepiece
  sentencepiece_model: $CVSS_ROOT/fr-en/fbank2unit/spm_unigram10000.model
global_cmvn:
  stats_npz_path: $CVSS_ROOT/fr-en/gcmvn.npz
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
  checkpoint: $CVSS_ROOT/fr-en/vocoder/mhubert_lyr11_km1000_en/g_00500000
  config: $CVSS_ROOT/fr-en/vocoder/mhubert_lyr11_km1000_en/config.json
  type: code_hifigan

vocab_filename_src: spm_unigram10000.txt
bpe_tokenizer_src:
  bpe: sentencepiece
  sentencepiece_model: $CVSS_ROOT/fr-en/fbank2unit/spm_unigram10000.model
```

For the ```src_fbank80.zip``` file, use [this script](https://github.com/ictnlp/NAST-S2x/blob/main/preprocessing/prep_fbank.py) to create it. Below, we provide a sample for its usage.
```
$covost2_data_root=yourpath
$cvssc_data_root=yourpath
$output_root=yourpath

python prep_fbank.py \
    --covost-data-root $covost2_data_root \
    --cvss-data-root $cvssc_data_root \
    --output-root  $output_root \
    --cmvn-type global
```




