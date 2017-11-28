cd ./nips_router_bot/utils
for t in ~/teams/day$1/*; do cat ~/export/day$1/alice_bob_ds.json | ./leaderboard.py ${t}; done


