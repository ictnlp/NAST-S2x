CVSS_ROOT=path_to_your_data
PATH=path_to_your_model  #This is the path to your checkpoint
CHUNK_SIZE=32  #This is an example, should be modified according to your model
OUT_ROOT=path_to_your_output
NAST_DIR=path_to_nast_dir #This is the path of our provided nast as a plugin to Fairseq

python nast/cli/generate_ctc_unit.py ${CVSS_ROOT} \
  --user-dir ${NAST_DIR} \
  --config-yaml config.yaml --gen-subset test --task nat_speech_to_unit_ctc_modified --src-upsample-ratio 1 --unit-size 2 --hidden-upsample-ratio 6 --main-context ${CHUNK_SIZE} --right-context ${CHUNK_SIZE} \
  --path ${PATH} \
  --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 \
  --target-is-code --target-code-size 1000 \
  --batch-size 1 --beam 1 --scoring sacrebleu > ${OUT_ROOT}
