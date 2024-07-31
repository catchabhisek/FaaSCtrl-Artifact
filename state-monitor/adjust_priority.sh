PRIORITY_BS=$1
AFFINITY_BS=$2

PRIORITY_MR=$3
AFFINITY_MR=$4

PRIORITY_OD=$5
AFFINITY_OD=$6

PRIORITY_SA=$7
AFFINITY_SA=$8

PRIORITY_EG=$9
AFFINITY_EG=${10}

AFFINITY_RS=${11}

LS_IMAGES="binary_alert|markdown-to-html|squeezenet|stock_analysis|email_gen"

cont_id=( $(docker ps --no-trunc | grep -E binary_alert | awk '{print $1}' | grep -v CONTAINER) )
for id in "${cont_id[@]}"
do
    cid=$(ps -ef | grep ${id} | grep containerd-shim | grep -v grep | awk '{print $2}')
    ppid=$(ps -ef | grep ${cid} | grep -v containerd-shim | grep -v perf | grep -v grep | grep -v init | awk '{print $2}')
    pid=$(ps -ef | grep ${ppid} | grep python | grep -v perf | grep -v bash| grep -v grep | awk '{print $2}')

    if [[ -z "$pid" ]]; then
        continue
    fi

    if [[ "$PRIORITY_BS" != "-1" ]]
    then
        sudo chrt -a -r -p ${PRIORITY_BS} ${cid}
        sudo chrt -a -r -p ${PRIORITY_BS} ${ppid}
        sudo chrt -a -r -p ${PRIORITY_BS} ${pid}
    else
        sudo chrt -a -o -p 0 ${cid}
        sudo chrt -a -o -p 0 ${ppid}
        sudo chrt -a -o -p 0 ${pid}
        echo "sudo chrt -a -o -p 0 ${pid}"
    fi

    if [[ "$AFFINITY_BS" != "-1" ]]
    then
        sudo taskset -a -cp ${AFFINITY_BS} ${cid} 1> /dev/null 2> /dev/null
        sudo taskset -a -cp ${AFFINITY_BS}  ${ppid} 1> /dev/null 2> /dev/null
        sudo taskset -a -cp ${AFFINITY_BS}  ${pid} 1> /dev/null 2> /dev/null
    else
        sudo taskset -a -cp ${AFFINITY_RS}  ${cid} 1> /dev/null 2> /dev/null
        sudo taskset -a -cp ${AFFINITY_RS}  ${ppid} 1> /dev/null 2> /dev/null
        sudo taskset -a -cp ${AFFINITY_RS}  ${pid} 1> /dev/null 2> /dev/null
    fi
done

cont_id=( $(docker ps --no-trunc | grep -E markdown-to-html | awk '{print $1}' | grep -v CONTAINER) )
for id in "${cont_id[@]}"
do
    cid=$(ps -ef | grep ${id} | grep containerd-shim | grep -v grep | awk '{print $2}')
    ppid=$(ps -ef | grep ${cid} | grep -v containerd-shim | grep -v perf | grep -v grep | grep -v init | awk '{print $2}')
    pid=$(ps -ef | grep ${ppid} | grep python | grep -v perf | grep -v bash| grep -v grep | awk '{print $2}')

    if [[ -z "$pid" ]]; then
        continue
    fi

    if [[ "$PRIORITY_MR" != "-1" ]]
    then
        sudo chrt -a -r -p ${PRIORITY_MR} ${cid}
        sudo chrt -a -r -p ${PRIORITY_MR} ${ppid}
        sudo chrt -a -r -p ${PRIORITY_MR} ${pid}
    else
        sudo chrt -a -o -p 0 ${cid}
        sudo chrt -a -o -p 0 ${ppid}
        sudo chrt -a -o -p 0 ${pid}
        echo "sudo chrt -a -o -p 0 ${pid}"
    fi

    if [[ "$AFFINITY_MR" != "-1" ]]
    then
        sudo taskset -a -cp ${AFFINITY_MR} ${cid} 1> /dev/null 2> /dev/null
        sudo taskset -a -cp ${AFFINITY_MR}  ${ppid} 1> /dev/null 2> /dev/null
        sudo taskset -a -cp ${AFFINITY_MR}  ${pid} 1> /dev/null 2> /dev/null
    else
        sudo taskset -a -cp ${AFFINITY_RS}  ${cid} 1> /dev/null 2> /dev/null
        sudo taskset -a -cp ${AFFINITY_RS}  ${ppid} 1> /dev/null 2> /dev/null
        sudo taskset -a -cp ${AFFINITY_RS}  ${pid} 1> /dev/null 2> /dev/null
    fi
done

cont_id=( $(docker ps --no-trunc | grep -E squeezenet | awk '{print $1}' | grep -v CONTAINER) )
for id in "${cont_id[@]}"
do
    cid=$(ps -ef | grep ${id} | grep containerd-shim | grep -v grep | awk '{print $2}')
    ppid=$(ps -ef | grep ${cid} | grep -v containerd-shim | grep -v perf | grep -v grep | grep -v init | awk '{print $2}')
    pid=$(ps -ef | grep ${ppid} | grep python | grep -v perf | grep -v bash| grep -v grep | awk '{print $2}')

    if [[ -z "$pid" ]]; then
        continue
    fi

    if [[ "$PRIORITY_OD" != "-1" ]]
    then
        sudo chrt -a -r -p ${PRIORITY_OD} ${cid}
        sudo chrt -a -r -p ${PRIORITY_OD} ${ppid}
        sudo chrt -a -r -p ${PRIORITY_OD} ${pid}
    else
        sudo chrt -a -o -p 0 ${cid}
        sudo chrt -a -o -p 0 ${ppid}
        sudo chrt -a -o -p 0 ${pid}
        echo "sudo chrt -a -o -p 0 ${pid}"
    fi

    if [[ "$AFFINITY_OD" != "-1" ]]
    then
        sudo taskset -a -cp ${AFFINITY_OD} ${cid} 1> /dev/null 2> /dev/null
        sudo taskset -a -cp ${AFFINITY_OD}  ${ppid} 1> /dev/null 2> /dev/null
        sudo taskset -a -cp ${AFFINITY_OD}  ${pid} 1> /dev/null 2> /dev/null
    else
        sudo taskset -a -cp 0-31  ${cid} 1> /dev/null 2> /dev/null
        sudo taskset -a -cp 0-31  ${ppid} 1> /dev/null 2> /dev/null
        sudo taskset -a -cp 0-31  ${pid} 1> /dev/null 2> /dev/null
    fi
done

cont_id=( $(docker ps --no-trunc | grep -E stock_analysis | awk '{print $1}' | grep -v CONTAINER) )
for id in "${cont_id[@]}"
do
    cid=$(ps -ef | grep ${id} | grep containerd-shim | grep -v grep | awk '{print $2}')
    ppid=$(ps -ef | grep ${cid} | grep -v containerd-shim | grep -v perf | grep -v grep | grep -v init | awk '{print $2}')
    pid=$(ps -ef | grep ${ppid} | grep python | grep -v perf | grep -v bash| grep -v grep | awk '{print $2}')

    if [[ -z "$pid" ]]; then
        continue
    fi

    if [[ "$PRIORITY_SA" != "-1" ]]
    then
        sudo chrt -a -r -p ${PRIORITY_SA} ${cid}
        sudo chrt -a -r -p ${PRIORITY_SA} ${ppid}
        sudo chrt -a -r -p ${PRIORITY_SA} ${pid}
    else
        sudo chrt -a -o -p 0 ${cid}
        sudo chrt -a -o -p 0 ${ppid}
        sudo chrt -a -o -p 0 ${pid}
        echo "sudo chrt -a -o -p 0 ${pid}"
    fi

    if [[ "$AFFINITY_SA" != "-1" ]]
    then
        sudo taskset -a -cp ${AFFINITY_SA} ${cid} 1> /dev/null 2> /dev/null
        sudo taskset -a -cp ${AFFINITY_SA}  ${ppid} 1> /dev/null 2> /dev/null
        sudo taskset -a -cp ${AFFINITY_SA}  ${pid} 1> /dev/null 2> /dev/null
    else
        sudo taskset -a -cp ${AFFINITY_RS}  ${cid} 1> /dev/null 2> /dev/null
        sudo taskset -a -cp ${AFFINITY_RS}  ${ppid} 1> /dev/null 2> /dev/null
        sudo taskset -a -cp ${AFFINITY_RS}  ${pid} 1> /dev/null 2> /dev/null
    fi
done

cont_id=( $(docker ps --no-trunc | grep -E email_gen | awk '{print $1}' | grep -v CONTAINER) )
for id in "${cont_id[@]}"
do
    cid=$(ps -ef | grep ${id} | grep containerd-shim | grep -v grep | awk '{print $2}')
    ppid=$(ps -ef | grep ${cid} | grep -v containerd-shim | grep -v perf | grep -v grep | grep -v init | awk '{print $2}')
    pid=$(ps -ef | grep ${ppid} | grep python | grep -v perf | grep -v bash| grep -v grep | awk '{print $2}')

    if [[ -z "$pid" ]]; then
        continue
    fi

    if [[ "$PRIORITY_EG" != "-1" ]]
    then
        sudo chrt -a -r -p ${PRIORITY_EG} ${cid}
        sudo chrt -a -r -p ${PRIORITY_EG} ${ppid}
        sudo chrt -a -r -p ${PRIORITY_EG} ${pid}
    else
        sudo chrt -a -o -p 0 ${cid}
        sudo chrt -a -o -p 0 ${ppid}
        sudo chrt -a -o -p 0 ${pid}
        echo "sudo chrt -a -o -p 0 ${pid}"
    fi

    if [[ "$AFFINITY_EG" != "-1" ]]
    then
        sudo taskset -a -cp ${AFFINITY_EG} ${cid} 1> /dev/null 2> /dev/null
        sudo taskset -a -cp ${AFFINITY_EG}  ${ppid} 1> /dev/null 2> /dev/null
        sudo taskset -a -cp ${AFFINITY_EG}  ${pid} 1> /dev/null 2> /dev/null
    else
        sudo taskset -a -cp ${AFFINITY_RS}  ${cid} 1> /dev/null 2> /dev/null
        sudo taskset -a -cp ${AFFINITY_RS}  ${ppid} 1> /dev/null 2> /dev/null
        sudo taskset -a -cp ${AFFINITY_RS}  ${pid} 1> /dev/null 2> /dev/null
    fi
done
