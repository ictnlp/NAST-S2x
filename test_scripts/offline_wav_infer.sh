CVSS_ROOT=path_to_your_data

IN_ROOT=path_to_your_input  #This is the path of the output of scripts ``offline_s2u_infer.sh``
OUT_ROOT=path_to_your_output

VOCODER_CKPT=${CVSS_ROOT}/vocoder/mhubert_lyr11_km1000_en/g_00500000
VOCODER_CFG=${CVSS_ROOT}/vocoder/mhubert_lyr11_km1000_en/config.json

grep "^Unit\-" ${IN_ROOT} | \
  sed 's/^Unit-//ig' | sort -nk1 | cut -f3 \
  > ${IN_ROOT}.sort

python fairseq/examples/speech_to_speech/generate_waveform_from_code.py \
  --in-code-file ${IN_ROOT}.sort \
  --vocoder ${VOCODER_CKPT} --vocoder-cfg {$VOCODER_CFG} \
  --results-path ${OUT_ROOT} --dur-prediction