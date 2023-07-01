#!/usr/bin/env bash

#reference:
#https://blog.csdn.net/baye_DOA/article/details/105781704
#
#data url:
#https://github.com/jayaram1125/Single-Word-Speech-Recognition-using-GMM-HMM-
#
# sh run.sh --stage -1 --stop_stage 5 --system_version windows


system_version=windows
verbose=true;

stage=-1
stop_stage=3

work_dir="$(pwd)"
dict_file="dict.txt"


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


file_dir="${work_dir}/data"


export PYTHONPATH="${work_dir}/../.."


if [ $system_version == "windows" ]; then
  #source /data/local/bin/OpenASR/bin/activate
  alias python3='C:/Users/tianx/PycharmProjects/virtualenv/OpenASR/Scripts/python.exe'
elif [ $system_version == "centos" ] || [ $system_version == "ubuntu" ]; then
  alias python3='/data/local/bin/OpenASR/bin/python3'
fi


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  $verbose && echo "stage -1: download wav"

  mkdir "${file_dir}"
  cd "${file_dir}" || exit 1;

  git clone https://github.com/jayaram1125/Single-Word-Speech-Recognition-using-GMM-HMM-
  tar -zxvf "Single-Word-Speech-Recognition-using-GMM-HMM-/audio.tar.gz"
  rm -rf "Single-Word-Speech-Recognition-using-GMM-HMM-"
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  $verbose && echo "stage 0: prepare data"

  cd "${work_dir}" || exit 1;
  python3 1.prepare_data.py \
  --file_dir "${file_dir}" \
  --dict_file "${dict_file}" \

fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  $verbose && echo "stage 1: train model"

  cd "${work_dir}" || exit 1;
  python3 2.train_model.py \
  --file_dir "${file_dir}"

fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  $verbose && echo "stage 2: test model"

  cd "${work_dir}" || exit 1;
  python3 3.test_model.py \
  --file_dir "${file_dir}" \
  --dict_file "${dict_file}" \

fi

