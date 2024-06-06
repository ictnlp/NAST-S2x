VOCODER_CKPT=path_to_your/mhubert_lyr11_km1000_en/g_00500000
VOCODER_CFG=path_to_your/mhubert_lyr11_km1000_en/config.json

DATA_PATH=
CHUNK_SIZE=
CKPT_PATH=
OUT_ROOT=path_to_your_output

SEGMENT_SIZE=${CHUNK_SIZE}0

simuleval \
    --data-bin ${DATA_PATH} \
    --source ${DATA_PATH}/test.wav_list --target ${DATA_PATH}/test.en \
    --model-path ${CKPT_PATH} \
    --config-yaml config.yaml --target-speech-lang en \
    --agent nast/agents/nast_speech2speech_agent_s2s.py \
    --wait-until 0 --main-context ${CHUNK_SIZE} --right-context ${CHUNK_SIZE} --sample-rate 48000 \
    --output ${OUT_ROOT} \
    --source-segment-size ${SEGMENT_SIZE} \
    --vocoder ${VOCODER_CKPT} --vocoder-cfg ${VOCODER_CFG} --dur-prediction \
    --quality-metrics ASR_BLEU  --latency-metrics AL LAAL AP DAL ATD NumChunks RTF StartOffset EndOffset \
    --device gpu --continue-unfinished