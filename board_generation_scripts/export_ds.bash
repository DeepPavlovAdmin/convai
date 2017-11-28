#!/usr/bin/env bash

#host=${1}
#shift
db=${1}
shift
#user=${1}
#shift
#password=${1}
#shift
days=${1}
shift

mongoexport --db ${db} --collection dialogs \
            --fields 'dialogId,users,context,evaluation,thread' \
            --query "$(./oid.py ${days})" --sort "{_id: 1}" 2>/dev/null

