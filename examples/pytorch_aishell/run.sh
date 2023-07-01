#!/usr/bin/env bash

# sh run.sh --stage -1 --stop_stage -1
# sh run.sh --stage 0 --stop_stage 0
# sh run.sh --stage 1 --stop_stage 1
# sh run.sh --stage 2 --stop_stage 2
# sh run.sh --stage 3 --stop_stage 3
# sh run.sh --stage -1 --stop_stage 1
#
# sh run.sh --stage 0 --stop_stage 0 --system_version windows --data_dir D:/programmer/asr_datasets/aishell
# sh run.sh --stage 3 --stop_stage 6 --system_version centos
# sh run.sh --stage -1 --stop_stage 3 --system_version centos


# params
system_version="windows";
verbose=true;

stage=0 # start from 0 if you need to start from data preparation
stop_stage=5

train_subset=train.json
valid_subset=valid.json
test_subset=test.json
global_cmvn=global_cmvn
vocabulary=vocabulary

patience=5

work_dir="$(pwd)"

train_config="${work_dir}/conf/train_conformer.jsonnet"


# parse options
while true; do
  [ -z "${1:-}" ] && break;  # break if there are no arguments
  case "$1" in
    --*) name=$(echo "$1" | sed s/^--// | sed s/-/_/g);
      eval '[ -z "${'"$name"'+xxx}" ]' && echo "$0: invalid option $1" 1>&2 && exit 1;
      old_value="(eval echo \\$$name)";
      if [ "${old_value}" == "true" ] || [ "${old_value}" == "false" ]; then
        was_bool=true;
      else
        was_bool=false;
      fi

      # Set the variable to the right value-- the escaped quotes make it work if
      # the option had spaces, like --cmd "queue.pl -sync y"
      eval "${name}=\"$2\"";

      # Check that Boolean-valued arguments are really Boolean.
      if $was_bool && [[ "$2" != "true" && "$2" != "false" ]]; then
        echo "$0: expected \"true\" or \"false\": $1 $2" 1>&2
        exit 1;
      fi
      shift 2;
      ;;

    *) break;
  esac
done


file_dir="${work_dir}/file_dir"
data_dir="/data/tianxing/PycharmProjects/datasets/aishell"
serialization_dir="serialization_dir"

mkdir -p "${file_dir}"
mkdir -p "${data_dir}"

export PYTHONPATH="${work_dir}/../.."


if [ $system_version == "windows" ]; then
  #source /data/local/bin/OpenASR/bin/activate
  alias python3='C:/Users/tianx/PycharmProjects/virtualenv/OpenASR/Scripts/python.exe'
elif [ $system_version == "centos" ] || [ $system_version == "ubuntu" ]; then
  alias python3='/data/local/bin/OpenASR/bin/python3'
fi


function search_best_ckpt() {
  version="$1";
  patience="$2";

  cd "${file_dir}/${serialization_dir}" || exit 1
  last_epoch=$(ls "lightning_logs/${version}/checkpoints" | \
               grep ckpt | awk -F'[=-]' '/epoch/ {print$2}' | \
               sort -n | awk 'END {print}')
  target_epoch=$((last_epoch - patience))
  target_file=null
  for file in $(ls "lightning_logs/${version}/checkpoints" | grep ckpt | sort -r):
  do
    this_epoch=$(echo "${file}" | awk -F'[=-]' '/epoch/ {print$2}');

    if [ "${this_epoch}" -le "${target_epoch}" ]; then
      target_file="${file}";
      break;
    fi
  done
  if [ "${target_file}" == null ]; then
    echo "no appropriate file found" && exit 1;
    return 0;
  fi
  echo "${target_file}"
}


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  $verbose && echo "stage -1: download and untar"

  cd "${data_dir}" || exit 1;
  data_aishell_size=$(/bin/ls -l data_aishell.tgz | awk '{print $5}')
  if [ ! -e data_aishell.tgz ] || [ "${data_aishell_size}" != "15582913665" ]; then
    # rm data_aishell.tgz
    wget -c www.openslr.org/resources/33/data_aishell.tgz --no-check-certificate
  fi

  resource_aishell_size=$(/bin/ls -l resource_aishell.tgz | awk '{print $5}')
  if [ ! -e resource_aishell.tgz ] || [ "${resource_aishell_size}" != "1246920" ]; then
    # rm resource_aishell.tgz
    wget -c www.openslr.org/resources/33/resource_aishell.tgz --no-check-certificate
  fi

  if [ ! -d resource_aishell ]; then
    echo "un-tarring archive: resource_aishell.tgz"
    tar -zxvf resource_aishell.tgz
  fi

  if [ ! -d data_aishell ]; then
    echo "un-tarring archive: data_aishell.tgz"
    tar -zxvf data_aishell.tgz
    cd data_aishell/wav || exit 1;

    for wav in ./*.tar.gz; do
        echo "Extracting wav from ${wav}"
        tar -zxf ${wav} && rm ${wav}
    done
  fi
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  $verbose && echo "stage 0: prepare data"
  cd "${work_dir}" || exit 1;

  python3 1.prepare_data.py \
  --aishell_audio_dir "${data_dir}/data_aishell/wav" \
  --aishell_text "${data_dir}/data_aishell/transcript/aishell_transcript_v0.8.txt" \
  --train_subset "${file_dir}/${train_subset}" \
  --valid_subset "${file_dir}/${valid_subset}" \
  --test_subset "${file_dir}/${test_subset}" \

fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  $verbose && echo "stage 1: compute cmvn stats"
  cd "${work_dir}" || exit 1;

  python3 2.compute_cmvn_stats.py \
  --train_subset "${file_dir}/${train_subset}" \
  --output_cmvn "${file_dir}/${global_cmvn}" \

fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  $verbose && echo "stage 2: make vocabulary"
  cd "${work_dir}" || exit 1;

  python3 3.make_vocabulary.py \
  --train_subset "${file_dir}/${train_subset}" \
  --vocabulary "${file_dir}/${vocabulary}" \

fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  $verbose && echo "stage 3: train model"
  cd "${work_dir}" || exit 1;

  python3 4.train_model.py \
  --train_subset "${file_dir}/${train_subset}" \
  --vocabulary "${file_dir}/${vocabulary}" \
  --train_config "${train_config}" \

fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  $verbose && echo "stage 4: export model"

  target_file=$(search_best_ckpt version_0 "${patience}");
  test target_file || exit 1;

  cd "${work_dir}" || exit 1;

  python3 5.export_models.py \
  --train_config "${train_config}" \
  --ckpt_path "${file_dir}/${serialization_dir}/lightning_logs/version_0/checkpoints/${target_file}" \
  --state_dict_filename "${file_dir}/pytorch_model.bin" \

fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  $verbose && echo "stage 5: evaluation"

  cd "${work_dir}" || exit 1;

  for subset in ${train_subset} ${valid_subset} ${test_subset}
  do
    python3 6.evaluation.py \
    --train_config "${train_config}" \
    --state_dict_filename "${file_dir}/pytorch_model.bin" \
    --dataset "${file_dir}/${subset}" \
    --output_file "${file_dir}/eval_${subset}"
  done

fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  $verbose && echo "stage 6: compute wer"

  cd "${work_dir}" || exit 1;

  for subset in ${train_subset} ${valid_subset} ${test_subset}
  do
    filename=${subset%.*}
    basename=${filename##*/}

    python3 7.compute_wer.py \
    --eval_file "${file_dir}/eval_${subset}" > "${file_dir}/wer_${basename##*/}.txt"
  done

fi


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  $verbose && echo "stage 7: test model"

  cd "${work_dir}" || exit 1;

  for subset in ${train_subset} ${valid_subset} ${test_subset}
  do
    python3 8.test_model.py \
    --train_config "${train_config}" \
    --state_dict_filename "${file_dir}/pytorch_model.bin"

  done

fi
