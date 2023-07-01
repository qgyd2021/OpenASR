#!/usr/bin/env bash
# 安装 cmake
# cmake 下载
# https://cmake.org/download/
# https://cmake.org/files/


# 参数:
cmake_version=3.25.0
system_version="centos";


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

echo "system_version: ${system_version}";

if [ ${system_version} = "centos" ]; then
  yum install -y wget

  mkdir -p /data/dep
  cd /data/dep || exit 1;
  if [ ! -e cmake-${cmake_version}-linux-x86_64.tar.gz ]; then
    # echo 3.25.0 | awk -F '.' '{print $1 "." $2}'
    cmake_version_folder="$(echo ${cmake_version} | awk -F '.' '{print $1 "." $2}')";
    # https://cmake.org/files/v3.25
    # https://cmake.org/files/v3.25/cmake-3.25.0-linux-x86_64.tar.gz
    url="https://cmake.org/files/v${cmake_version_folder}/cmake-${cmake_version}-linux-x86_64.tar.gz"
    wget -P /data/dep -c "${url}"
  fi

  cd /data/dep || exit 1;
  if [ ! -d cmake-${cmake_version}-linux-x86_64 ]; then
    tar -zxvf cmake-${cmake_version}-linux-x86_64.tar.gz
    cd cmake-${cmake_version}-linux-x86_64 || exit 1;
  fi

  if [ "$(cmake --version | grep "cmake version" | awk -F ' ' '{print $NF}')" != "${cmake_version}" ]; then
    echo "cmake ${cmake_version} already exist" || exit 0;
  fi

  # usr/bin/cmake
  /data/dep/cmake-3.25.0-linux-x86_64/bin/cmake --version
  # cmake version 3.25.0

  # 软链接
  rm -rf /usr/bin/cmake
  rm -rf /usr/bin/ccmake
  ln -s /data/dep/cmake-3.25.0-linux-x86_64/bin/cmake /usr/bin/cmake
  ln -s /data/dep/cmake-3.25.0-linux-x86_64/bin/ccmake /usr/bin/ccmake

  cmake --version
  ccmake --version
fi
