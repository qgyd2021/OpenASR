#!/usr/bin/env bash

# sh run.sh --stage -1 --stop_stage 4 --system_version windows
# sh run.sh --stage 4 --stop_stage 4 --system_version windows


system_version=windows
verbose=true;

stage=-1
stop_stage=3

train_subset=train.json
global_cmvn=global_cmvn
vocabulary=dict.txt

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

  cd "${file_dir}" || exit 1
  last_epoch=$(ls "lightning_logs/${version}/checkpoints" | \
               grep ckpt | awk 'END {print}' | \
               awk -F'[=-]' '/epoch/ {print$2}')
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
  $verbose && echo "stage -1: download data"

  mkdir "${file_dir}"
  cd "${file_dir}" || exit 1;

  git clone https://github.com/nicedi/Chinese_YesNo_ASR
  mv Chinese_YesNo_ASR/yesno_cn yesno_cn
  rm -rf Chinese_YesNo_ASR

fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  $verbose && echo "stage 0: prepare data"
  cd "${work_dir}" || exit 1;

  python3 1.prepare_data.py \
  --file_dir "${file_dir}" \
  --yesno_cn_dir yesno_cn \
  --metadata train_list.txt \
  --train_subset "${train_subset}" \

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
  --train_subset "${train_subset}" \
  --vocabulary "${vocabulary}" \

fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 3 ]; then
  $verbose && echo "stage 4: test model"
  cd "${work_dir}" || exit 1;

  target_file=$(search_best_ckpt version_0 "${patience}");
  test target_file || exit 1;

  python3 5.test_model.py \
  --file_dir "${file_dir}" \
  --train_subset "${train_subset}" \
  --vocabulary "${vocabulary}" \
  --ckpt_path "lightning_logs/version_0/checkpoints/${target_file}" \

fi
