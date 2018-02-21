#!/usr/bin/env bash

script_home=bot_code
script_name="$script_home/bot.py"
pid_file="$script_home/bot.pid"

# returns a boolean and optionally the pid
running() {
    local status=false
    if [[ -f $pid_file ]]; then
        # check to see it corresponds to the running script
        local pid=$(< "$pid_file")
        local cmdline=/proc/$pid/cmdline
        # you may need to adjust the regexp in the grep command
        if [[ -f $cmdline ]] && grep -q "$script_name" $cmdline; then
            status="true $pid"
        fi
    fi
    echo $status
}

start() {
    echo "starting services"
    #if [ ! `docker inspect -f {{.State.Running}} bidaf 2>/dev/null` ]; then
    #    docker run -d --name bidaf -p 1995:1995 sld3/bi-att-flow:0.1.0
    #fi
    #if [ ! `docker inspect -f {{.State.Running}} corenlp 2>/dev/null` ]; then
    #    docker run -d --name corenlp -p 9000:9000 sld3/corenlp:3.6.0
    #fi

    #local opennmt_run_cmd="cd /root/opennmt && th tools/translation_server.lua \
    #  -host 0.0.0.0 -port 5556  -model /root/data/model.t7 -beam_size 12"
    #if [ ! `docker inspect -f {{.State.Running}} opennmt 2>/dev/null` ]; then
    #    docker run -d --name opennmt -it -p 5556:5556 -v $(pwd)/data:/root/data sld3/opennmt bash -c "$opennmt_run_cmd"
    #fi


    echo "starting $script_name"
    #python --version
    #nohup python $script_name &
    python $script_name &
    echo $! > "$pid_file"
}

stop() {
    kill $1 2>/dev/null
    rm "$pid_file"
    echo "stopped"
}

stopall() {
    kill $1 2>/dev/null
    rm "$pid_file"

    #echo "stopping services"
    #docker stop bidaf  #QA
    #docker rm bidaf    #QA
    #docker stop opennmt
    #docker rm opennmt
    #docker stop corenlp
    #docker rm corenlp

    #echo "stopped"
}

read running pid < <(running)

case $1 in
    start)
        if $running; then
            echo "$script_name is already running with PID $pid"
        else
            start
        fi
        ;;
    stop)
        stop $pid
        ;;
    restart)
        stop $pid
        start
        ;;
    restartall)
        stopall $pid
        start
        ;;
    status)
        if $running; then
            echo "$script_name is running with PID $pid"
        else
            echo "$script_name is not running"
        fi
        ;;
    *)  echo "usage: $0 <start|stop|restart|status>"
        echo "stop is killing the bot only, while stopall shuts down all the services with it"
        exit
        ;;
esac
