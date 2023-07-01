#!/usr/bin/env bash



# params
system_version="windows";
verbose=true;
stage=0 # start from 0 if you need to start from data preparation
stop_stage=5

train_subset=train.json
global_cmvn=global_cmvn
vocabulary=dict.txt
ciempiess_dataset_file=ciempiess_LDC2015S07.tgz
model_config=config.jsonnet

work_dir="$(pwd)"


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
ciempiess_dataset_dir="/data/tianxing/PycharmProjects/datasets/ciempiess"

export PYTHONPATH="${work_dir}/../.."


if [ $system_version == "windows" ]; then
  #source /data/local/bin/OpenASR/bin/activate
  alias python3='C:/Users/tianx/PycharmProjects/virtualenv/OpenASR/Scripts/python.exe'
elif [ $system_version == "centos" ] || [ $system_version == "ubuntu" ]; then
  alias python3='/data/local/bin/OpenASR/bin/python3'
fi


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  $verbose && echo "stage -1: download and untar"
  # https://ciempiess.org/downloads

  if [ ! -e "${ciempiess_dataset_file}" ]; then
    exit 1;
  fi

  if [ ! -d "${ciempiess_dataset_dir}" ]; then
    cd "${ciempiess_dataset_dir}" || exit 1;
    tar -zxvf "${ciempiess_dataset_file}"
  fi
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  $verbose && echo "stage 0: prepare data"
  cd "${work_dir}" || exit 1;

  python3 1.prepare_data.py \
  --file_dir "${file_dir}" \
  --ciempiess_dataset_dir "${ciempiess_dataset_dir}" \

fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  $verbose && echo "stage 1: compute cmvn stats"
  cd "${work_dir}" || exit 1;

  python3 2.compute_cmvn_stats.py \
  --file_dir "${file_dir}" \
  --train_subset "${train_subset}" \
  --num_workers 1 \
  --output_cmvn "${global_cmvn}" \

fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  $verbose && echo "stage 2: make vocabulary"
  cd "${work_dir}" || exit 1;

  python3 3.make_vocabulary.py \
  --file_dir "${file_dir}" \
  --train_subset "${train_subset}" \
  --vocabulary "${vocabulary}" \

fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  $verbose && echo "stage 3: train model"
  cd "${work_dir}" || exit 1;

  python3 4.train_model.py \
  --file_dir "${file_dir}" \
  --config "${model_config}" \
  --train_subset "${train_subset}" \
  --vocabulary "${vocabulary}" \

fi
