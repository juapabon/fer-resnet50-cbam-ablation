#!/bin/bash
# Empaqueta los best.pth de cada grupo de ejecuciones terminadas y los sube al borrador de Zenodo (sin publicar).
# Uso: ZENODO_TOKEN=... subir_pesos.sh <bucket_url> grupo1 [grupo2 ...]   (el token sólo vive en el entorno)
B=$1; shift
W=$HOME/fer_frontiers_work; Z=$W/zenodo; mkdir -p $Z; L=$Z/subida.log
for g in "$@"; do
  f=$Z/weights_$g.tar
  [ -s $f ] || (cd $W && ls $g/*/meta.json | xargs -n1 dirname | sed 's|$|/best.pth|' | tar -cf $f -T -)
  echo "$(date '+%F %T') $g: $(tar -tf $f | wc -l) pesos, $(du -h $f | cut -f1)" >> $L
  code=$(curl -s -o $Z/resp_$g.json -w '%{http_code}' --retry 3 -H "Authorization: Bearer $ZENODO_TOKEN" --upload-file $f "$B/weights_$g.tar")
  echo "$(date '+%F %T') $g: HTTP $code $(python3 -c "import json;r=json.load(open('$Z/resp_$g.json'));print(r.get('size'),r.get('checksum'),r.get('message',''))")" >> $L
  [ "$code" = 200 ] || [ "$code" = 201 ] && rm -f $f
done
echo "$(date '+%F %T') FIN" >> $L
