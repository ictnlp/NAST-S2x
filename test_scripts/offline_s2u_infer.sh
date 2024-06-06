CVSS_ROOT=path_to_your_data
PATH=path_to_your_model
MAIN=chunk_size
OUT_ROOT=path_to_your_output
NAST_DIR=path_to_nast_dir

python nast/cli/generate_ctc_unit.py ${CVSS_ROOT}/fr-en/fbank2unit \
  --user-dir ${NAST_DIR} \
  --config-yaml config.yaml --gen-subset test --task nat_speech_to_unit_ctc_modified --src-upsample-ratio 1 --unit-size 2 --hidden-upsample-ratio 6 --main-context ${MAIN} --right-context ${MAIN} \
  --path ${PATH} \
  --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 \
  --target-is-code --target-code-size 1000 \
  --batch-size 1 --beam 1 --scoring sacrebleu > ${OUT_ROOT}
