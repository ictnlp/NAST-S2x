# NAST-S2X: A Fast and End-to-End Simultaneous Speech-to-Any Translation Model
[![arXiv](https://img.shields.io/badge/arXiv-2406.06937-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2406.06937)
[![model](https://img.shields.io/badge/%F0%9F%A4%97%20-NAST_S2X-blue.svg)](https://huggingface.co/ICTNLP/NAST-S2X)
<!--<p align="center">
<img src="https://github.com/ictnlp/NAST-S2x/assets/43530347/02d6dea6-5887-459e-9938-bc510b6c850c"/>  
</p> -->

## News🔥
*  **(2024/06/27) We have created a tutorial to guide you through preprocessing the data and running NAST-S2X on your machine. You can find it at this [URL](https://github.com/ictnlp/NAST-S2x/tree/main/preprocessing).**
*  **We have published our paper on arXiv, available at https://arxiv.org/abs/2406.06937.**
*  **We have released the checkpoints and datasets for reference at [Hugging Face🤗](https://huggingface.co/ICTNLP/NAST-S2X/tree/main).**

## Features
* 🤖 **An end-to-end model without intermediate text decoding**
* 💪 **Supports offline and streaming decoding of all modalities**
* ⚡️ **28× faster inference compared to autoregressive models**

## Examples
#### We present an example of French-to-English translation using chunk sizes of 320 ms, 2560 ms, and in offline conditions.
* Generation with chunk sizes of 320 ms and 2560 ms starts generating English translation before the source speech is complete.
* In the examples of simultaneous interpretation, the left audio channel is the input streaming speech, and the right audio channel is the simultaneous translation.
> [!NOTE]
> For a better experience, please wear headphones.
  
Chunk Size 320ms            |  Chunk Size 2560ms |  Offline
:-------------------------:|:-------------------------: |:-------------------------:
<video src="https://github.com/ictnlp/NAST-S2x/assets/43530347/52f2d5c4-43ad-49cb-844f-09575ef048e0" width="100"></video>  |  <video src="https://github.com/ictnlp/NAST-S2x/assets/43530347/56475dee-1649-40d9-9cb6-9fe033f6bb32"></video> | <video src="https://github.com/ictnlp/NAST-S2x/assets/43530347/b6fb1d09-b418-45f0-84e9-e6ed3a2cea48"></video>

Source Speech Transcript            |  Reference Text Translation
:-------------------------:|:-------------------------:
Avant la fusion des communes, Rouge-Thier faisait partie de la commune de Louveigné.| before the fusion of the towns rouge thier was a part of the town of louveigne

> [!NOTE]
> For more examples, please check https://nast-s2x.github.io/.

## Performance

* ⚡️ **Lightning Fast**: 28× faster inference and competitive quality in offline speech-to-speech translation
* 👩‍💼 **Simultaneous**: Achieves high-quality simultaneous interpretation within a delay of less than 3 seconds
* 🤖 **Unified Framework**: Support end-to-end text & speech generation in one model
 
  
**Check Details** 👇
  Offline-S2S          |  Simul-S2S   |  Simul-S2T
:-------------------------:|:-------------------------:|:-------------------------:
![image](https://github.com/ictnlp/NAST-S2x/assets/43530347/abf6931f-c6be-4870-8f58-3a338e3b2b5c)| ![image](https://github.com/ictnlp/NAST-S2x/assets/43530347/9a57bf02-c606-4a78-af3e-1c0d1f25d27e) | ![image](https://github.com/ictnlp/NAST-S2x/assets/43530347/6ecfe401-770c-4dc0-9c50-e76a8c20b84b)




## Architecture
<p align="center">
<img src="https://github.com/ictnlp/NAST-S2x/assets/43530347/404cdd56-a9d9-4c10-96aa-64f0c7605248" width="800" />  
</p>

* **Fully Non-autoregressive:** Trained with **CTC-based non-monotonic latent alignment loss [(Shao and Feng, 2022)](https://arxiv.org/abs/2210.03953)** and **glancing mechanism [(Qian et al., 2021)](https://arxiv.org/abs/2008.07905)**.
* **Minimum Human Design:** Seamlessly switch between offline translation and simultaneous interpretation **by adjusting the chunk size**.
* **End-to-End:** Generate target speech **without** target text decoding.

# Sources and Usage
## Model
> [!NOTE]
> We release French-to-English speech-to-speech translation models trained on the CVSS-C dataset to reproduce results in our paper. You can train models in your desired languages by following the instructions provided below.

[🤗 Model card](https://huggingface.co/ICTNLP/NAST-S2X)
| Chunk Size | checkpoint | ASR-BLEU | ASR-BLEU (Silence Removed) | Average Lagging                                                                             |
| ----------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------- |---------------------------------------------------------------- |
| 320ms    |  [checkpoint](https://huggingface.co/ICTNLP/NAST-S2X/blob/main/chunk_320ms.pt) | 19.67 |  24.90 | -393ms  |
| 1280ms    | [checkpoint](https://huggingface.co/ICTNLP/NAST-S2X/blob/main/chunk_1280ms.pt) | 20.20 | 25.71 | 3330ms |
| 2560ms    | [checkpoint](https://huggingface.co/ICTNLP/NAST-S2X/blob/main/chunk_2560ms.pt) | 24.88 | 26.14 |  4976ms  |
| Offline    | [checkpoint](https://huggingface.co/ICTNLP/NAST-S2X/blob/main/Offline.pt) | 25.82 | -  | -   |

| Vocoder |
| --- |
| [checkpoint](https://huggingface.co/ICTNLP/NAST-S2X/tree/main/vocoder)|

## Inference
> [!WARNING]
> Before executing all the provided shell scripts, please ensure to replace the variables in the file with the paths specific to your machine.

### Offline Inference
* **Data preprocessing**: Follow the instructions in the [document](https://github.com/ictnlp/NAST-S2x/tree/main/preprocessing).
* **Generate Acoustic Unit**: Execute [``offline_s2u_infer.sh``](https://github.com/ictnlp/NAST-S2x/blob/main/test_scripts/offline_s2u_infer.sh)
* **Generate Waveform**: Execute [``offline_wav_infer.sh``](https://github.com/ictnlp/NAST-S2x/blob/main/test_scripts/offline_wav_infer.sh)
* **Evaluation**: Using Fairseq's [ASR-BLEU evaluation toolkit](https://github.com/facebookresearch/fairseq/tree/main/examples/speech_to_speech/asr_bleu)
### Simultaneous Inference
* We use our customized fork of [``SimulEval: b43a7c``](https://github.com/Paulmzr/SimulEval/tree/b43a7c7a9f20bb4c2ff48cf1bc573b4752d7081e) to evaluate the model in simultaneous inference.  This repository is built upon the official [``SimulEval: a1435b``](https://github.com/facebookresearch/SimulEval/tree/a1435b65331cac9d62ea8047fe3344153d7e7dac) and includes additional latency scorers.
* **Data preprocessing**: Follow the instructions in the [document](https://github.com/ictnlp/NAST-S2x/tree/main/preprocessing).
* **Streaming Generation and Evaluation**: Execute [``streaming_infer.sh``](https://github.com/ictnlp/NAST-S2x/blob/main/test_scripts/streaming_infer.sh)

## Train your own NAST-S2X
* **Data preprocessing**: Follow the instructions in the [document](https://github.com/ictnlp/NAST-S2x/tree/main/preprocessing).
* **Encoder Pretraining**: Execute [``pretrain_encoder.sh``](https://github.com/ictnlp/NAST-S2x/blob/main/train_scripts/pretrain_encoder.sh)
* **CTC Pretraining**: Execute [``train_ctc.sh``](https://github.com/ictnlp/NAST-S2x/blob/main/train_scripts/train_ctc.sh)
* **NMLA Training**: Execute [``train_nmla.sh``](https://github.com/ictnlp/NAST-S2x/blob/main/train_scripts/train_nmla.sh)

## Citing

Please kindly cite us if you find our papers or codes useful.

```
@inproceedings{
ma2024nonautoregressive,
title={A Non-autoregressive Generation Framework for End-to-End Simultaneous Speech-to-Any Translation},
author={Ma, Zhengrui and Fang, Qingkai and Zhang, Shaolei and Guo, Shoutao and Feng, Yang and Zhang, Min
},
booktitle={Proceedings of ACL 2024},
year={2024},
}
```

```
@inproceedings{
fang2024ctcs2ut,
title={CTC-based Non-autoregressive Textless Speech-to-Speech Translation},
author={Fang, Qingkai and Ma, Zhengrui and Zhou, Yan and Zhang, Min and Feng, Yang
},
booktitle={Findings of ACL 2024},
year={2024},
}
```




