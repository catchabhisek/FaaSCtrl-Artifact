FAAS_ROOT="/home/abhisek/Serverless-SLA/faas-profiler"
DIR=${FAAS_ROOT}"/state"

TEST_DURATION=$1
EXP_NAME=$2

LS_DOCKER_PERF_PREFIX=$DIR"/ls_perflog_"
LD_DOCKER_PERF_PREFIX=$DIR"/ld_perflog_"
TIME_PREFIX=$DIR"/perfsystime_"
LD_APPS=$DIR"/ld_apps_info"

echo "Cleaning previous saved results"
rm -rf $LS_DOCKER_PERF_PREFIX*
rm -rf $LD_DOCKER_PERF_PREFIX*
rm -rf $LD_APPS*
rm -rf $TIME_PREFIX*

LS_IMAGES="binary_alert|markdown-to-html|squeezenet|stock_analysis|email_gen"

id_list=()
pid_list=()

LS_PERF_EVENTS=$(cat ${FAAS_ROOT}/state-monitor/serverless-hpe)
LD_PERF_EVENTS=$(cat ${FAAS_ROOT}/state-monitor/serverless-hpe)

EXP_DIR=${FAAS_ROOT}"/results/"${EXP_NAME}

mkdir -p $EXP_DIR

echo "$( date +%s%N | cut -b1-13 )" > ${LD_APPS}

while :
do
    cont_id=( $(docker ps --no-trunc | grep -E binary_alert | awk '{print $1}' | grep -v CONTAINER) )

    for id in "${cont_id[@]}"
    do
        if [[ -z "$id" ]]; then
                continue
        fi

        if ! [[ ${id_list[@]} =~ (^|[[:space:]])${id}($|[[:space:]]) ]]
        then
            sleep 1
            cid=$(ps -ef | grep ${id} | grep containerd-shim | grep -v grep | awk '{print $2}')
            ppid=$(ps -ef | grep ${cid} | grep -v containerd-shim | grep -v perf | grep -v grep | grep -v init | awk '{print $2}')
            pid=$(ps -ef | grep ${ppid} | grep python | grep -v perf | grep -v bash| grep -v init |grep -v grep | awk '{print $2}')

            if [[ -z "$pid" ]]; then
                continue
            fi

            sudo perf stat -e ${LS_PERF_EVENTS} -I 2000 -p ${pid},${cid},${ppid} >>${LS_DOCKER_PERF_PREFIX}_bs_${id}_${pid} 2>&1 &

            echo "Container ID: {$id}"
            echo "PID: ${pid}"

            id_list+=(${id})
            pid_list+=(${id},${pid},"bs")
        fi
    done

    cont_id=( $(docker ps --no-trunc | grep -E markdown-to-html | awk '{print $1}' | grep -v CONTAINER) )

    for id in "${cont_id[@]}"
    do
        if [[ -z "$id" ]]; then
                continue
        fi

        if ! [[ ${id_list[@]} =~ (^|[[:space:]])${id}($|[[:space:]]) ]]
        then
            sleep 1
            cid=$(ps -ef | grep ${id} | grep containerd-shim | grep -v grep | awk '{print $2}')
            ppid=$(ps -ef | grep ${cid} | grep -v containerd-shim | grep -v perf | grep -v grep | grep -v init | awk '{print $2}')
            pid=$(ps -ef | grep ${ppid} | grep python | grep -v perf | grep -v bash| grep -v init |grep -v grep | awk '{print $2}')

            if [[ -z "$pid" ]]; then
                continue
            fi

            sudo perf stat -e ${LS_PERF_EVENTS} -I 2000 -p ${pid},${cid},${ppid} >>${LS_DOCKER_PERF_PREFIX}_mr_${id}_${pid} 2>&1 &

            echo "Container ID: {$id}"
            echo "PID: ${pid}"

            id_list+=(${id})
            pid_list+=(${id},${pid},"mr")
        fi
    done

    cont_id=( $(docker ps --no-trunc | grep -E squeezenet | awk '{print $1}' | grep -v CONTAINER) )

    for id in "${cont_id[@]}"
    do
        if [[ -z "$id" ]]; then
                continue
        fi

        if ! [[ ${id_list[@]} =~ (^|[[:space:]])${id}($|[[:space:]]) ]]
        then
            sleep 1
            cid=$(ps -ef | grep ${id} | grep containerd-shim | grep -v grep | awk '{print $2}')
            ppid=$(ps -ef | grep ${cid} | grep -v containerd-shim | grep -v perf | grep -v grep | grep -v init | awk '{print $2}')
            pid=$(ps -ef | grep ${ppid} | grep python | grep -v perf | grep -v bash| grep -v init |grep -v grep | awk '{print $2}')

            if [[ -z "$pid" ]]; then
                continue
            fi

            sudo perf stat -e ${LS_PERF_EVENTS} -I 2000 -p ${pid},${cid},${ppid} >>${LS_DOCKER_PERF_PREFIX}_od_${id}_${pid} 2>&1 &

            echo "Container ID: {$id}"
            echo "PID: ${pid}"

            id_list+=(${id})
            pid_list+=(${id},${pid},"od")
        fi
    done

    cont_id=( $(docker ps --no-trunc | grep -E stock_analysis | awk '{print $1}' | grep -v CONTAINER) )

    for id in "${cont_id[@]}"
    do
        if [[ -z "$id" ]]; then
                continue
        fi

        if ! [[ ${id_list[@]} =~ (^|[[:space:]])${id}($|[[:space:]]) ]]
        then
            sleep 1
            cid=$(ps -ef | grep ${id} | grep containerd-shim | grep -v grep | awk '{print $2}')
            ppid=$(ps -ef | grep ${cid} | grep -v containerd-shim | grep -v perf | grep -v grep | grep -v init | awk '{print $2}')
            pid=$(ps -ef | grep ${ppid} | grep python | grep -v perf | grep -v bash| grep -v init |grep -v grep | awk '{print $2}')

            if [[ -z "$pid" ]]; then
                continue
            fi

            sudo perf stat -e ${LS_PERF_EVENTS} -I 2000 -p ${pid},${cid},${ppid} >>${LS_DOCKER_PERF_PREFIX}_sa_${id}_${pid} 2>&1 &

            echo "Container ID: {$id}"
            echo "PID: ${pid}"

            id_list+=(${id})
            pid_list+=(${id},${pid},"sa")
        fi
    done


    cont_id=( $(docker ps --no-trunc | grep -E email_gen | awk '{print $1}' | grep -v CONTAINER) )

    for id in "${cont_id[@]}"
    do
        if [[ -z "$id" ]]; then
                continue
        fi

        if ! [[ ${id_list[@]} =~ (^|[[:space:]])${id}($|[[:space:]]) ]]
        then
            sleep 1
            cid=$(ps -ef | grep ${id} | grep containerd-shim | grep -v grep | awk '{print $2}')
            ppid=$(ps -ef | grep ${cid} | grep -v containerd-shim | grep -v perf | grep -v grep | grep -v init | awk '{print $2}')
            pid=$(ps -ef | grep ${ppid} | grep python | grep -v perf | grep -v bash| grep -v init |grep -v grep | awk '{print $2}')

            if [[ -z "$pid" ]]; then
                continue
            fi

            sudo perf stat -e ${LS_PERF_EVENTS} -I 2000 -p ${pid},${cid},${ppid} >>${LS_DOCKER_PERF_PREFIX}_eg_${id}_${pid} 2>&1 &

            echo "Container ID: {$id}"
            echo "PID: ${pid}"

            id_list+=(${id})
            pid_list+=(${id},${pid},"eg")
        fi
    done

    cont_id=( $(docker ps --no-trunc | grep -E pagerank | grep -v whisksystem_invokerHealthTestAction | awk '{print $1}' | grep -v CONTAINER) )

    for id in "${cont_id[@]}"
    do
        if ! [[ ${id_list[@]} =~ (^|[[:space:]])${id}($|[[:space:]]) ]]
        then
            sleep 1
            cid=$(ps -ef | grep ${id} | grep containerd-shim | grep -v grep | awk '{print $2}')
            ppid=$(ps -ef | grep ${cid} | grep -v containerd-shim | grep -v perf | grep -v grep | grep -v init | awk '{print $2}')
            pid=$(ps -ef | grep ${ppid} | grep -E "python|node" | grep -v perf | grep -v sh| grep -v grep | awk '{print $2}')

            if [[ -z "$pid" ]]; then
                continue
            fi

            sudo perf stat -e ${LD_PERF_EVENTS} -I 2000 -p ${pid},${cid},${ppid} >>${LD_DOCKER_PERF_PREFIX}_pg_${id}_${pid} 2>&1 &
            id_list+=(${id})
            pid_list+=(${id},${pid},"pg")
        fi
    done

    cont_id=( $(docker ps --no-trunc | grep -E dna_visualization | grep -v whisksystem_invokerHealthTestAction | awk '{print $1}' | grep -v CONTAINER) )

    for id in "${cont_id[@]}"
    do
        if ! [[ ${id_list[@]} =~ (^|[[:space:]])${id}($|[[:space:]]) ]]
        then
            sleep 1
            cid=$(ps -ef | grep ${id} | grep containerd-shim | grep -v grep | awk '{print $2}')
            ppid=$(ps -ef | grep ${cid} | grep -v containerd-shim | grep -v perf | grep -v grep | grep -v init | awk '{print $2}')
            pid=$(ps -ef | grep ${ppid} | grep -E "python|node" | grep -v perf | grep -v sh| grep -v grep | awk '{print $2}')

            if [[ -z "$pid" ]]; then
                continue
            fi

            sudo perf stat -e ${LD_PERF_EVENTS} -I 2000 -p ${pid},${cid},${ppid} >>${LD_DOCKER_PERF_PREFIX}_dv_${id}_${pid} 2>&1 &
            id_list+=(${id})
            pid_list+=(${id},${pid},"dv")
        fi
    done

    cont_id=( $(docker ps --no-trunc | grep -E review_analysis | grep -v whisksystem_invokerHealthTestAction | awk '{print $1}' | grep -v CONTAINER) )

    for id in "${cont_id[@]}"
    do
        if ! [[ ${id_list[@]} =~ (^|[[:space:]])${id}($|[[:space:]]) ]]
        then
            sleep 1
            cid=$(ps -ef | grep ${id} | grep containerd-shim | grep -v grep | awk '{print $2}')
            ppid=$(ps -ef | grep ${cid} | grep -v containerd-shim | grep -v perf | grep -v grep | grep -v init | awk '{print $2}')
            pid=$(ps -ef | grep ${ppid} | grep -E "python|node" | grep -v perf | grep -v sh| grep -v grep | awk '{print $2}')

            if [[ -z "$pid" ]]; then
                continue
            fi

            sudo perf stat -e ${LD_PERF_EVENTS} -I 2000 -p ${pid},${cid},${ppid} >>${LD_DOCKER_PERF_PREFIX}_ra_${id}_${pid} 2>&1 &
            id_list+=(${id})
            pid_list+=(${id},${pid},"ra")
        fi
    done

    cont_id=( $(docker ps --no-trunc | grep -E video_processing | grep -v whisksystem_invokerHealthTestAction | awk '{print $1}' | grep -v CONTAINER) )

    for id in "${cont_id[@]}"
    do
        if ! [[ ${id_list[@]} =~ (^|[[:space:]])${id}($|[[:space:]]) ]]
        then
            sleep 1
            cid=$(ps -ef | grep ${id} | grep containerd-shim | grep -v grep | awk '{print $2}')
            ppid=$(ps -ef | grep ${cid} | grep -v containerd-shim | grep -v perf | grep -v grep | grep -v init | awk '{print $2}')
            pid=$(ps -ef | grep ${ppid} | grep -E "python|node" | grep -v perf | grep -v sh| grep -v grep | awk '{print $2}')

            if [[ -z "$pid" ]]; then
                continue
            fi

            sudo perf stat -e ${LD_PERF_EVENTS} -I 2000 -p ${pid},${cid},${ppid} >>${LD_DOCKER_PERF_PREFIX}_vp_${id}_${pid} 2>&1 &
            id_list+=(${id})
            pid_list+=(${id},${pid},"vp")
        fi
    done

    cont_id=( $(docker ps --no-trunc | grep -E action-nodejs | grep -v whisksystem_invokerHealthTestAction | awk '{print $1}' | grep -v CONTAINER) )

    for id in "${cont_id[@]}"
    do
        if ! [[ ${id_list[@]} =~ (^|[[:space:]])${id}($|[[:space:]]) ]]
        then
            sleep 1
            cid=$(ps -ef | grep ${id} | grep containerd-shim | grep -v grep | awk '{print $2}')
            ppid=$(ps -ef | grep ${cid} | grep -v containerd-shim | grep -v perf | grep -v grep | grep -v init | awk '{print $2}')
            pid=$(ps -ef | grep ${ppid} | grep -E "python|node" | grep -v perf | grep -v sh| grep -v grep | awk '{print $2}')

            if [[ -z "$pid" ]]; then
                continue
            fi

            sudo perf stat -e ${LD_PERF_EVENTS} -I 2000 -p ${pid},${cid},${ppid} >>${LD_DOCKER_PERF_PREFIX}_ir_${id} 2>&1 &
            id_list+=(${id})
            pid_list+=(${id},${pid},"ir")
        fi
    done

    deleted_pids=()
    deleted_ids=()


    # Note that $(NF-X) stores the CPU wait time, #nvcs, #vcs, sys time, usr time etc. These values are not originally part of the /proc filesystem. We had made a slight change to the kernel. You can use schedstats and other available /proc features to collect the CPU wait time and #nvcs used in this paper.

    for id in "${pid_list[@]}"
    do
        inps=(${id//,/ })
       
        if ! [ -f "/proc/${inps[1]}/stat" ]; then
            deleted_pids+=(${id})
            deleted_ids+=(${inps[0]})
            continue
        fi

        if  [ -z "$inps[1]" ]; then
            deleted_pids+=(${id})
            deleted_ids+=(${inps[0]})
            continue
        fi

        echo "${inps[1]} $( cat /proc/${inps[1]}/stat | awk '{print $14, $15, $(NF-7), $(NF-6), $(NF-5), $(NF-4), $(NF-3), $(NF-2), $(NF-1), $NF}' ) $(date +"%a %b %d %H:%M:%S.%6N %Y")" >> ${TIME_PREFIX}_${inps[2]}_${inps[0]}_${inps[1]}
    done

    for id in "${deleted_pids[@]}"
    do
        pid_list=${pid_list[@]/$id}
    done

    # for id in "${deleted_ids[@]}"
    # do
    #     echo "Removing the id ${id}"
    #     id_list=${id_list[@]/$id}
    # done

done