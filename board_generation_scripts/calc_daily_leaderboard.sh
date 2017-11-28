cd ./nips_router_bot/utils
echo "Export raw dataset for leaderboard"
./export_ds.bash convai-bot 0 > ../../daily_leaderboard_ds.json
echo "Calc user leaderboard"
cat ../../daily_leaderboard_ds.json | ./user_leaderboard.py > ../../user_leaderboard.csv
echo "Calc bot leaderboard"
cat ../../daily_leaderboard_ds.json | ./bot_leaderboard.py > ../../bot_leaderboard.csv
 
