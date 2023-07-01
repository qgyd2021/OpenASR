#!/usr/bin/env bash
# 参考官方实现, 但要改造成自己容易理解的形式: thirdparty/wenet-2.1.0/examples/aishell/s0/run.sh

# sh run.sh --stage -1 --stop_stage -1
# sh run.sh --stage 0 --stop_stage 0
# sh run.sh --stage 1 --stop_stage 1
# sh run.sh --stage 2 --stop_stage 2
# sh run.sh --stage 3 --stop_stage 3
# sh run.sh --stage -1 --stop_stage 1
#
# sh run.sh --stage 0 --stop_stage 0 --system_version windows --aishell_dataset_dir D:/programmer/asr_datasets/aishell
#



#export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export CUDA_VISIBLE_DEVICES="0"

# params
system_version="windows";
verbose=true;
stage=0 # start from 0 if you need to start from data preparation
stop_stage=5

# The num of machines(nodes) for multi-machine training, 1 is for one machine.
# NFS is required if num_nodes > 1.
num_nodes=1

# The rank of each node or machine, which ranges from 0 to `num_nodes - 1`.
# You should set the node_rank=0 on the first machine, set the node_rank=1
# on the second machine, and so on.
node_rank=0

checkpoint=

# Optional train_config
# 1. conf/train_transformer.yaml: Standard transformer
# 2. conf/train_conformer.yaml: Standard conformer
# 3. conf/train_unified_conformer.yaml: Unified dynamic chunk causal conformer
# 4. conf/train_unified_transformer.yaml: Unified dynamic chunk transformer
# 5. conf/train_u2++_conformer.yaml: U2++ conformer
# 6. conf/train_u2++_transformer.yaml: U2++ transformer
train_config=conf/train_conformer.yaml
train_set=train

dict=data/dict/lang_char.txt

work_dir="$(pwd)"
file_dir="${work_dir}/file_dir"

aishell_dataset_dir="${file_dir}/aishell"


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
  $verbose && echo "stage -1: download and untar"

  mkdir -p "${aishell_dataset_dir}"
  cd "${aishell_dataset_dir}" || exit 1;
  data_aishell_size=$(/bin/ls -l data_aishell.tgz | awk '{print $5}')
  if [ ! -e data_aishell.tgz ] || [ "${data_aishell_size}" != "15582913665" ]; then
    rm data_aishell.tgz
    wget www.openslr.org/resources/33/data_aishell.tgz --no-check-certificate
  fi

  resource_aishell_size=$(/bin/ls -l resource_aishell.tgz | awk '{print $5}')
  if [ ! -e resource_aishell.tgz ] || [ "${resource_aishell_size}" != "1246920" ]; then
    rm resource_aishell.tgz
    wget www.openslr.org/resources/33/resource_aishell.tgz --no-check-certificate
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
  $verbose && echo "stage 0: data preparation"
  python3 1.aishell_data_prepare.py \
  --data_dir "${file_dir}/data" \
  --aishell_audio_dir "${aishell_dataset_dir}/data_aishell/wav" \
  --aishell_text "${aishell_dataset_dir}/data_aishell/transcript/aishell_transcript_v0.8.txt"

fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  $verbose && echo "stage 1: compute cmvn stats"
  python3 2.compute_cmvn_stats.py \
  --num_workers 4 \
  --train_config $train_config \
  --in_scp "${file_dir}/data/${train_set}/wav.scp" \
  --out_cmvn "${file_dir}/data/$train_set/global_cmvn"

fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  $verbose && echo "stage 2: make a dictionary"
  python3 3.make_dictionary.py \
  --dict_filename "${file_dir}/${dict}" \
  --train_data_text "${file_dir}/data/${train_set}/text" \

fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  $verbose && echo "prepare data, prepare required format"
  for split in "dev" "test" "${train_set}"; do
    python3 4.prepare_data.py \
    --wav_file "${file_dir}/data/${split}/wav.scp" \
    --text_file "${file_dir}/data/${split}/text" \
    --output_file "${file_dir}/data/${split}/data.list"

  done
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  $verbose && echo "train model"

  mkdir -p "${file_dir}"
  # You have to rm `INIT_FILE` manually when you resume or restart a
  # multi-machine training.
  INIT_FILE="${file_dir}/ddp_init"
  init_method=file://$(readlink -f "${INIT_FILE}")
  echo "$0: init method is ${init_method}"
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  dist_backend="gloo"
  world_size=$(( "${num_gpus}"\*${num_nodes} ))

  echo "total gpus is: ${world_size}"

  cp data/${train_set}/global_cmvn "${file_dir}"
  cmvn_opts="--cmvn ${file_dir}/global_cmvn"

  # train.py rewrite $train_config to $dir/train.yaml with model input
  # and output dimension, and $dir/train.yaml will be used for inference
  # and export.
  for ((i = 0; i < num_gpus; ++i)); do
  {
    gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$(( i+1 )))

    # Rank of each gpu/process used for knowing whether it is
    # the master of a worker.
    rank=$(( ${node_rank}\*$num_gpus+i ))

    python wenet/bin/train.py --gpu $gpu_id \
      --config $train_config \
      --data_type "raw" \
      --symbol_table $dict \
      --train_data data/$train_set/data.list \
      --cv_data data/dev/data.list \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir "${file_dir}" \
      --ddp.init_method "${init_method}" \
      --ddp.world_size $world_size \
      --ddp.rank $rank \
      --ddp.dist_backend $dist_backend \
      --num_workers 1 \
      "${cmvn_opts}" \
      --pin_memory
  } &
  done
  wait
fi
