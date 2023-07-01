#!/usr/bin/env bash

# sh run.sh --stage -1 --stop_stage -1
# sh run.sh --stage 0 --stop_stage 0
# sh run.sh --stage 1 --stop_stage 1
# sh run.sh --stage 2 --stop_stage 2
# sh run.sh --stage 3 --stop_stage 3
# sh run.sh --stage -1 --stop_stage 1
#
# sh run.sh --stage 0 --stop_stage 0 --system_version windows --aishell_dataset_dir D:/programmer/asr_datasets/aishell
#

system_version="windows";
verbose=true;
stage=0 # start from 0 if you need to start from data preparation
stop_stage=5


work_dir="$(pwd)"
file_dir="${work_dir}/file_dir"
yesno_dataset_dir="${file_dir}/yesno"


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


if [ $system_version == "windows" ]; then
  #source /data/local/bin/OpenASR/bin/activate
  alias python3='C:/Users/tianx/PycharmProjects/virtualenv/OpenASR/Scripts/python.exe'
elif [ $system_version == "centos" ] || [ $system_version == "ubuntu" ]; then
  alias python3='/data/local/bin/OpenASR/bin/python3'
fi


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  $verbose && echo "stage -1: download data"

  mkdir -p "${yesno_dataset_dir}" && cd "${yesno_dataset_dir}" || exit 1;

  if [ ! -f waves_yesno.tar.gz ]; then
    wget http://www.openslr.org/resources/1/waves_yesno.tar.gz || exit 1;
    # was:
    # wget http://sourceforge.net/projects/kaldi/files/waves_yesno.tar.gz || exit 1;
  fi

  if [ ! -d waves_yesno ]; then
    tar -xvzf waves_yesno.tar.gz || exit 1;
  fi
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  $verbose && echo "stage 0: prepare data"

  mkdir -p "${file_dir}" && cd "${work_dir}" || exit 1;

  python3 1.prepare_data.py \
  --file_dir "${file_dir}" \
  --yesno_dir "${yesno_dataset_dir}/waves_yesno"

fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  $verbose && echo "stage 1: make dictionary"

  mkdir -p "${file_dir}" && cd "${work_dir}" || exit 1;

  python3 2.make_dictionary.py \
  --file_dir "${file_dir}"

fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  $verbose && echo "stage 2: train model"

  mkdir -p "${file_dir}" && cd "${work_dir}" || exit 1;

  python3 3.train_model.py \
  --file_dir "${file_dir}"

fi
