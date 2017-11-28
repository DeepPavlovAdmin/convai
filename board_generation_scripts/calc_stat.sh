cd ./nips_router_bot/utils
echo "Export raw dataset for stat"
./export_ds.bash convai-bot $1 > ../../stat/day$1_ds.json
echo "Calc stat"
cat ../../stat/day$1_ds.json | ./stat.py > ../../stat/day$1.txt


