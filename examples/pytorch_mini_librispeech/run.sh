#!/usr/bin/env bash

# sh run.sh --stage -1 --stop_stage 4 --system_version centos
# sh run.sh --stage 3 --stop_stage 4 --system_version centos

# sh run.sh --stage -1 --stop_stage 4 --system_version windows
# sh run.sh --stage 4 --stop_stage 4 --system_version windows



system_version=windows
verbose=true;

stage=-1
stop_stage=3

train_subset=train.json
valid_subset=valid.json
global_cmvn=global_cmvn
vocabulary=vocabulary

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
data_dir="${file_dir}"

mkdir -p "${file_dir}"
mkdir -p "${data_dir}"

export PYTHONPATH="${work_dir}/../.."


if [ $system_version == "windows" ]; then
  #source /data/local/bin/OpenASR/bin/activate
  alias python3='C:/Users/tianx/PycharmProjects/virtualenv/OpenASR/Scripts/python.exe'
elif [ $system_version == "centos" ] || [ $system_version == "ubuntu" ]; then
  alias python3='/data/local/bin/OpenASR/bin/python3'
fi


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  $verbose && echo "stage -1: download data"
  # https://www.openslr.org/31/

  cd "${data_dir}" || exit 1;

  wget -c https://us.openslr.org/resources/31/train-clean-5.tar.gz --no-check-certificate
  wget -c https://us.openslr.org/resources/31/dev-clean-2.tar.gz --no-check-certificate

  if [ ! -d train-clean-5 ]; then
    tar -zxvf train-clean-5.tar.gz;
  fi

  if [ ! -d dev-clean-2 ]; then
    tar -zxvf dev-clean-2.tar.gz;
  fi
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  $verbose && echo "stage 0: prepare data"
  cd "${work_dir}" || exit 1;

  python3 1.prepare_data.py \
  --train_asr_data_dir "${data_dir}/LibriSpeech/train-clean-5" \
  --valid_asr_data_dir "${data_dir}/LibriSpeech/dev-clean-2" \
  --train_subset "${file_dir}/${train_subset}" \
  --valid_subset "${file_dir}/${valid_subset}"

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
