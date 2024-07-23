# Function to print usage
usage() {
  echo "Usage: $0 -p dir1 dir2 dir3 -d int1 int2 int3"
  exit 1
}

# Check if no arguments were passed
if [ $# -eq 0 ]; then
  usage
fi

# Initialize arrays to hold directories and integers
dirs=()
ints=()
versions=()
epochs=()
tf=false
o=false
name=""


# Parse options
while getopts ":p:d:v:e:i:to" opt; do
  case ${opt} in
    p)
      # Capture directory arguments 
      ((OPTIND--))
      while [[ ${OPTIND} -le $# && ! ${!OPTIND} =~ ^- ]]; do
        dirs+=("${!OPTIND}")
        ((OPTIND++))
      done
      ;;
    d)
      ((OPTIND--))
      while [[ ${OPTIND} -le $# && ! ${!OPTIND} =~ ^- ]]; do
        ints+=("${!OPTIND}")
        ((OPTIND++))
      done
      ;;
    v)
      ((OPTIND--))
      while [[ ${OPTIND} -le $# && ! ${!OPTIND} =~ ^- ]]; do
        versions+=("${!OPTIND}")
        ((OPTIND++))
      done
      ;;
    e)
      ((OPTIND--))
      while [[ ${OPTIND} -le $# && ! ${!OPTIND} =~ ^- ]]; do
        epochs+=("${!OPTIND}")
        ((OPTIND++))
      done
      ;;
    t)
	tf=true
	;;
    o)
	o=true
	;;
    i)
	name=${OPTARG}
	;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      usage
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      usage
      ;;
  esac
done

# Check if the number of directories and integers match
if [ ${#dirs[@]} -ne ${#ints[@]} ]; then
  echo "The number of directories and integers must match."
  usage
fi

# Process each directory and its associated integer
for ((i=0; i<${#dirs[@]}; i++)); do
  dir=${dirs[$i]}
  int=${ints[$i]}
  vers=${versions[$i]}
  screen -dmS "$name$i"
  if "$tf"; then
      if "$o"; then
	  echo "Executing: python3 multi_run.py -config_path $dir -device $int -model $vers -me ${epochs[@]} --tf --overfit"
	  screen -S "$name$i" -X exec python3 multi_run.py -config_path $dir -device $int -model $vers -me ${epochs[@]} --tf --overfit
      else
	  echo "Executing: python3 multi_run.py -config_path $dir -device $int -model $vers -me ${epochs[@]} --tf"
	  screen -S "$name$i" -X exec python3 multi_run.py -config_path $dir -device $int -model $vers -me ${epochs[@]} --tf
      fi
  else
      if "$o"; then
	  echo "Executing: python3 multi_run.py -config_path $dir -device $int -model $vers -me ${epochs[@]} --overfit"
	  screen -S "$name$i" -X exec python3 multi_run.py -config_path $dir -device $int -model $vers -me ${epochs[@]} --overfit
      else
	  echo "Executing: python3 multi_run.py -config_path $dir -device $int -model $vers -me ${epochs[@]}"
	  screen -S "$name$i" -X exec python3 multi_run.py -config_path $dir -device $int -model $vers -me ${epochs[@]}
      fi
  fi
done
