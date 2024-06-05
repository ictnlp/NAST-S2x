# NAST-S2X: A Fast and End-to-End Simultaneous Speech-to-Any Translation Model
<p align="center">
<img src="https://github.com/ictnlp/NAST-S2x/assets/43530347/02d6dea6-5887-459e-9938-bc510b6c850c"/>  
</p>

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
* 👩‍💼 **Simultaneous Decoding**: Support end-to-end text & speech generation in a unified framework
 
  
**Check Details** 👇
  Offline-S2S          |  Simul-S2S   |  Simul-S2T
:-------------------------:|:-------------------------:|:-------------------------:
![image](https://github.com/ictnlp/NAST-S2x/assets/43530347/abf6931f-c6be-4870-8f58-3a338e3b2b5c)| ![image](https://github.com/ictnlp/NAST-S2x/assets/43530347/631ecc44-748a-4ac9-bd1a-2d8563336a1c) | ![image](https://github.com/ictnlp/NAST-S2x/assets/43530347/adcbaedf-7740-405e-a139-2ab2fec63481)



## Architecture
<p align="center">
<img src="https://github.com/ictnlp/NAST-S2x/assets/43530347/404cdd56-a9d9-4c10-96aa-64f0c7605248" width="800" />  
</p>

* **Fully Non-autoregressive:** Trained with **CTC-based non-monotonic latent alignment loss [(Shao and Feng, 2022)](https://arxiv.org/abs/2210.03953)** and **glancing mechanism [(Qian et al., 2021)](https://arxiv.org/abs/2008.07905)**.
* **Minimum Human Design:** Seamlessly switch between offline translation and simultaneous interpretation **by adjusting the chunk size**.
* **End-to-End:** Generate target speech **without** target text decoding.

## Sources









